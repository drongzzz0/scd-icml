from .types import ImportPlan, FullPrefixImportPlan, tokens, integer, strict_construct
from .identity import sha256
from ..errors import IdentityMismatch, NoReusablePrefix, UnsupportedProfile

def get_import_policy(plan):
    return plan.import_policy if isinstance(plan,FullPrefixImportPlan) else 'blocktail'

def import_plan_from_data(data):
    if not isinstance(data,dict):raise IdentityMismatch('import plan must be an object')
    value=dict(data);value['suffix_ids']=tuple(value['suffix_ids'])
    cls=FullPrefixImportPlan if 'import_policy' in value else ImportPlan
    return strict_construct(cls,value)

def check_fullprefix_scope(profile,capability=None,*,resident=False):
    attention=profile.target.attention
    kinds=set(attention.get('layer_types',()))
    window=attention.get('sliding_window')
    if kinds!={'full'} or window is not None:
        from .capability import check_capability
        from .types import EngineCapability
        if (resident is not True or profile.method_id!='icml_reuse_patch' or profile.reader.kind!='none' or kinds!={'sliding'}
                or type(window) is not int or window<1 or not isinstance(capability,EngineCapability)
                or capability.backend!='FLASH_ATTN'
                or type(capability.max_model_len) is not int or not 0<capability.max_model_len<=window):
            raise UnsupportedProfile('fullprefix candidate supports only full attention or bounded original ICML SWA')
        check_capability(profile,capability)
    if capability is not None and (capability.kv_groups!=1 or capability.hma
            or (capability.prefix_cache and not capability.request_prefix_cache_policy)
            or capability.chunked_prefill or capability.tp!=1 or capability.pp!=1 or capability.block_size!=16):
        raise UnsupportedProfile('fullprefix candidate requires single-group serial block16 without APC/HMA/chunking')
    if window is not None and capability.prefix_cache:
        raise UnsupportedProfile('bounded SWA fullprefix requires APC off')

def allocated_prefix_slots(block_ids,block_size,plan):
    """Map exactly the approved tokens, including a partial final private block."""
    integer(block_size,'block_size');integer(plan.original_prefix_tokens,'original prefix tokens')
    integer(plan.imported_tokens,'imported tokens')
    ids=tuple(block_ids)
    for value in ids:integer(value,'allocated block ID',minimum=0)
    if not ids or len(set(ids))!=len(ids) or plan.imported_tokens>plan.original_prefix_tokens or plan.imported_tokens>len(ids)*block_size:
        raise IdentityMismatch('allocated prefix slots are empty, repeated or insufficient')
    return [ids[i//block_size]*block_size+i%block_size for i in range(plan.imported_tokens)]

def plan_import(prefix,prompt_ids,block_size,*,import_policy='blocktail'):
    integer(block_size,'block_size')
    prompt=tokens(prompt_ids); n=len(prefix.token_ids)
    if len(prompt)<n or prompt[:n]!=prefix.token_ids: raise IdentityMismatch('request does not match artifact prefix')
    if import_policy=='fullprefix':
        if block_size!=16:raise UnsupportedProfile('fullprefix candidate requires physical block16')
        if n==0 or len(prompt)<=n:raise IdentityMismatch('fullprefix requires a nonempty prefix and real following suffix')
        return FullPrefixImportPlan(n,n,0,prompt[n:],sha256({'schema':'scd-fullprefix-v1',
            'import_policy':'fullprefix','prompt_ids':prompt,'prefix':prefix,'block_size':block_size}))
    if import_policy!='blocktail':raise ValueError('unknown import policy')
    q=(min(n,len(prompt)-1)//block_size)*block_size
    if q==0: raise NoReusablePrefix('no complete importable block; native fallback must be explicit')
    return ImportPlan(n,q,n-q,prompt[q:],sha256({'prompt_ids':prompt,'prefix':prefix,'block_size':block_size}))
