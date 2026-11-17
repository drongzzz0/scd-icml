def combine_metrics(mode,preparation,generation):
    if mode not in ('E','H'):raise ValueError('mode must be E or H')
    source=float(preparation.get('source_prefill_ms',0.))
    if mode=='H' and source!=0:raise ValueError('H cannot hide recomputed source cost')
    startup=float(preparation.get('source_load_ms',0.))
    preparation_ms=source+float(preparation.get('convert_ms',0.))+float(preparation.get('publish_ms',0.))
    before=preparation_ms+float(generation.get('admission_ms',0.))+float(generation.get('consumer_queue_ms',0.))+float(generation.get('reader_before_ms',0.))
    first=generation.get('first_token_ms')
    timing_complete=generation.get('fixed_reader_receipt') is None or all(k in generation for k in ('reader_before_ms','reader_after_ms'))
    total=preparation_ms+float(generation['consume_wall_ms']) if 'consume_wall_ms' in generation else before+float(generation.get('generate_wall_ms',0.))+float(generation.get('reader_after_ms',0.))
    fallback=bool(generation.get('native_fallback',False))
    return {**preparation,**generation,'mode':mode,'timing_complete':timing_complete,'pipeline_first_token_ms':None if first is None or not timing_complete else before+float(first),'including_source_setup_first_token_ms':None if first is None or not timing_complete else startup+before+float(first),'pipeline_generation_ms':total if timing_complete else None,'network_latency_ms':None,'network_wire_bytes':None,'reused_tokens':0 if fallback else int(generation.get('imported_tokens',0)),'native_fallback':fallback,'quality_established':False,'source_setup_included_separately':True}
