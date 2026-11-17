class SCDRuntimeError(Exception):
    """A failed request or invalid artifact; not a successful reuse result."""

class IdentityMismatch(SCDRuntimeError): pass
class NoReusablePrefix(SCDRuntimeError): pass
class UnsupportedProfile(SCDRuntimeError): pass
class MissingAsset(SCDRuntimeError): pass
class AssetMismatch(SCDRuntimeError): pass
class ArtifactCorrupt(SCDRuntimeError): pass
class StoreCapacityExceeded(SCDRuntimeError): pass
class ReaderMismatch(IdentityMismatch): pass
class ValidationIncomplete(SCDRuntimeError): pass
