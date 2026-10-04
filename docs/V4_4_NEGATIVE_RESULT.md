# HarmonicFold spectral v4.4 development result

Status: rejected development candidate; not part of the paper mainline.

The v4.4 candidate combined nested spectral subband views with a learned
temporal/spectral evidence-reliability gate. On Benchmark split 0, seed
20260929, the combined model scored 28.571%, 67.202% and 77.798% at 0.4, 0.8
and 1.2 s, below the frozen v4.1 model at 29.940%, 67.321% and 79.286%.

The no-attention isolation showed that the nested subband stem reduced 0.4-s
accuracy from 28.571% to 26.726%, while increasing 0.8 and 1.2 s from 63.571%
and 78.214% to 64.345% and 81.071%. The subband representation therefore
trades short-window evidence for long-window evidence on this fold. Adding the
gate did not preserve the long-window improvement.

The candidate is rejected because it adds 16,801 parameters, has close prior
art in dynamic time-frequency/reliability-gated EEG fusion, and does not beat
the frozen mainline. It remains recorded to prevent repeated experimentation.
