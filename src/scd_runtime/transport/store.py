from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
import sqlite3
import time
import uuid
from .manifest import describe,restore
from ..assets.manifest import file_sha256
from ..contracts.identity import canonical_bytes,sha256
from ..errors import ArtifactCorrupt,StoreCapacityExceeded,MissingAsset,IdentityMismatch

@dataclass(frozen=True)
class ArtifactLease:
    artifact_id: str
    request_id: str
    lease_id: str

class ArtifactStore:
    """Local filesystem + SQLite. Eviction and pinning share a writer lock.

    No TTL-based lease stealing. The caller releases only its own completed
    requests; crashed-instance recovery requires an explicit confirmed list.
    """
    def __init__(self,root,capacity_bytes):
        if type(capacity_bytes) is not int or capacity_bytes<262144:raise StoreCapacityExceeded('store needs at least 256 KiB including metadata')
        self.root=Path(root).resolve();self.capacity_bytes=capacity_bytes
        if str(self.root).startswith('\\\\'):raise ValueError('store requires a local filesystem')
        self.root.mkdir(parents=True,exist_ok=True)
        for name in ('objects','staging'):
            path=self.root/name;path.mkdir(exist_ok=True)
            if path.is_symlink() or path.resolve().parent!=self.root:raise ValueError('store directory must not be a link')
        self.metadata_budget=min(1024*1024,capacity_bytes//2)
        self.db=self.root/'index.sqlite3'
        if self.db.is_symlink():raise ValueError('database must not be a link')
        with self._transaction() as con:
            con.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value INTEGER NOT NULL)')
            row=con.execute("SELECT value FROM settings WHERE key='capacity'").fetchone()
            if row and row[0]!=capacity_bytes:raise ValueError('capacity differs from existing store')
            con.execute("INSERT OR IGNORE INTO settings VALUES ('capacity',?)",(capacity_bytes,))
            con.execute('CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, bytes INTEGER NOT NULL, accessed REAL NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, request TEXT UNIQUE NOT NULL, artifact TEXT NOT NULL REFERENCES artifacts(id))')
            # A scheduler can release its lease before the client reads output.
            # Keep the small worker receipt until that client consumes it.
            con.execute('CREATE TABLE IF NOT EXISTS injections (lease TEXT PRIMARY KEY, request TEXT NOT NULL, artifact TEXT NOT NULL, receipt TEXT NOT NULL)')

    @contextmanager
    def _transaction(self):
        con=sqlite3.connect(self.db,timeout=30)
        try:
            con.execute('PRAGMA foreign_keys=ON');con.execute('PRAGMA journal_mode=DELETE');con.execute('PRAGMA synchronous=FULL')
            page=con.execute('PRAGMA page_size').fetchone()[0]
            con.execute(f'PRAGMA max_page_count={max(10,self.metadata_budget//(3*page))}')
            con.execute('BEGIN IMMEDIATE')
            yield con
            con.commit()
        except sqlite3.DatabaseError as exc:
            con.rollback()
            if 'full' in str(exc).lower():raise StoreCapacityExceeded('metadata capacity exhausted') from exc
            raise
        except BaseException:
            con.rollback();raise
        finally:con.close()

    def _object(self,identifier):
        if not isinstance(identifier,str) or re.fullmatch('[0-9a-f]{64}',identifier) is None:raise MissingAsset('invalid artifact ID')
        path=self.root/'objects'/identifier
        if path.is_symlink() or not path.resolve().is_relative_to(self.root/'objects'):raise ArtifactCorrupt('artifact path invalid')
        return path

    def _used_data(self):
        return sum(p.stat().st_size for folder in ('objects','staging') for p in (self.root/folder).rglob('*') if p.is_file())

    def _make_room(self,con,requested):
        available=self.capacity_bytes-self.metadata_budget
        if requested>available:raise StoreCapacityExceeded('artifact exceeds store capacity')
        while self._used_data()+requested>available:
            row=con.execute('SELECT id FROM artifacts WHERE id NOT IN (SELECT artifact FROM leases) ORDER BY accessed LIMIT 1').fetchone()
            if not row:raise StoreCapacityExceeded('all remaining artifacts pinned or staging space occupied')
            path=self._object(row[0])
            # This path is a validated, content-addressed child owned by this store.
            if path.exists():shutil.rmtree(path)
            con.execute('DELETE FROM artifacts WHERE id=?',row)

    def publish(self,kv):
        from safetensors.torch import save_file
        metadata,tensors=describe(kv)
        raw_bytes=sum(t.numel()*t.element_size() for t in tensors.values())
        reserve=raw_bytes+len(canonical_bytes(metadata))+4096+512*len(tensors)
        stage=self.root/'staging'/uuid.uuid4().hex
        with self._transaction() as con:
            self._make_room(con,reserve)
            stage.mkdir()
            try:
                payload=stage/'kv.safetensors';save_file(tensors,str(payload))
                metadata.update(payload_sha256=file_sha256(payload),payload_bytes=payload.stat().st_size)
                encoded=canonical_bytes(metadata);identifier=sha256(metadata)
                (stage/'manifest.json').write_bytes(encoded)
                size=payload.stat().st_size+len(encoded)
                if size>reserve:raise StoreCapacityExceeded('serialization exceeded reserved space')
                row=con.execute('SELECT id FROM artifacts WHERE id=?',(identifier,)).fetchone()
                dest=self._object(identifier)
                if row:
                    self._read(identifier)
                    con.execute('UPDATE artifacts SET accessed=? WHERE id=?',(time.time(),identifier))
                else:
                    if dest.exists():raise ArtifactCorrupt('unindexed object requires explicit recovery')
                    stage.rename(dest)
                    con.execute('INSERT INTO artifacts VALUES (?,?,?)',(identifier,size,time.time()))
                return identifier
            finally:
                if stage.exists() and stage.resolve().parent==self.root/'staging':shutil.rmtree(stage)

    def _read(self,identifier):
        folder=self._object(identifier)
        try:
            manifest=folder/'manifest.json';payload=folder/'kv.safetensors'
            if manifest.is_symlink() or payload.is_symlink():raise ArtifactCorrupt('linked payload forbidden')
            data=json.loads(manifest.read_text(encoding='utf-8'))
            if sha256(data)!=identifier or payload.stat().st_size!=data['payload_bytes'] or file_sha256(payload)!=data['payload_sha256']:raise ArtifactCorrupt('artifact checksum mismatch')
            return data,payload
        except (OSError,ValueError,KeyError,TypeError) as exc:raise ArtifactCorrupt('artifact incomplete or corrupt') from exc

    def pin(self,artifact_id,request_id):
        return self.pin_with_metadata(artifact_id,request_id)[0]

    def pin_with_metadata(self,artifact_id,request_id):
        """Validate and pin atomically, returning metadata from that same read.

        Consumers still validate the payload again in load(). This avoids the
        duplicate scan when admission immediately needs the pinned metadata.
        """
        self._object(artifact_id)
        if not isinstance(request_id,str) or not request_id:raise ValueError('request ID required')
        with self._transaction() as con:
            if not con.execute('SELECT 1 FROM artifacts WHERE id=?',(artifact_id,)).fetchone():raise MissingAsset('artifact not READY')
            metadata,_=self._read(artifact_id)
            prior=con.execute('SELECT id,artifact FROM leases WHERE request=?',(request_id,)).fetchone()
            if prior:
                if prior[1]!=artifact_id:raise IdentityMismatch('request already pinned another artifact')
                return ArtifactLease(artifact_id,request_id,prior[0]),metadata
            identifier=uuid.uuid4().hex
            con.execute('INSERT INTO leases VALUES (?,?,?)',(identifier,request_id,artifact_id))
            return ArtifactLease(artifact_id,request_id,identifier),metadata

    def metadata(self,lease):
        with self._transaction() as con:
            self._check_lease(con,lease)
            return self._read(lease.artifact_id)[0]

    @staticmethod
    def _check_lease(con,lease):
        if not con.execute('SELECT 1 FROM leases WHERE id=? AND request=? AND artifact=?',(lease.lease_id,lease.request_id,lease.artifact_id)).fetchone():raise MissingAsset('lease is not active')

    def load(self,lease,plan):
        from safetensors.torch import load_file
        with self._transaction() as con:
            self._check_lease(con,lease);data,payload=self._read(lease.artifact_id)
            kv=restore(data,load_file(str(payload),device='cpu'))
            if plan.original_prefix_tokens!=len(kv.prefix.token_ids) or not 0<plan.imported_tokens<=plan.original_prefix_tokens:raise IdentityMismatch('import plan does not match payload')
            con.execute('UPDATE artifacts SET accessed=? WHERE id=?',(time.time(),lease.artifact_id))
            return kv

    def release(self,lease):
        with self._transaction() as con:
            con.execute('DELETE FROM leases WHERE id=? AND request=? AND artifact=?',(lease.lease_id,lease.request_id,lease.artifact_id))

    def record_injection(self,lease,receipt):
        with self._transaction() as con:
            self._check_lease(con,lease)
            con.execute('INSERT OR REPLACE INTO injections VALUES (?,?,?,?)',
                        (lease.lease_id,lease.request_id,lease.artifact_id,json.dumps(receipt,sort_keys=True)))

    def pop_injection(self,lease):
        with self._transaction() as con:
            row=con.execute('SELECT receipt FROM injections WHERE lease=? AND request=? AND artifact=?',
                            (lease.lease_id,lease.request_id,lease.artifact_id)).fetchone()
            if row is None:raise MissingAsset('worker injection receipt missing; cannot claim KV reuse')
            con.execute('DELETE FROM injections WHERE lease=?',(lease.lease_id,))
            return json.loads(row[0])

    def discard_injection(self,lease):
        with self._transaction() as con:
            con.execute('DELETE FROM injections WHERE lease=? AND request=? AND artifact=?',
                        (lease.lease_id,lease.request_id,lease.artifact_id))

    def reclaim_finished(self,request_ids):
        """Caller supplies only requests confirmed dead, never arbitrary TTL expiry."""
        with self._transaction() as con:
            for request_id in request_ids:
                con.execute('DELETE FROM leases WHERE request=?',(request_id,))
                con.execute('DELETE FROM injections WHERE request=?',(request_id,))
