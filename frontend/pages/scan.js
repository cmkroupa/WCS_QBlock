import Link from 'next/link';
import Script from 'next/script';
import { useEffect, useRef, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001';

function scoreClass(prediction) {
  const p = String(prediction || '').toLowerCase();
  if (p === 'phishing') return 'is-phishing';
  if (p === 'benign' || p === 'safe' || p === 'legitimate') return 'is-safe';
  return 'is-unknown';
}

function scorePct(score) {
  return `${Math.round(Math.max(0, Math.min(1, Number(score) || 0)) * 100)}%`;
}

export default function ScanPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);

  const [status, setStatus] = useState('booting'); // booting | scanning | loading | result | error
  const [detectedUrl, setDetectedUrl] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const stopCamera = () => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  const processUrl = async (url) => {
    setDetectedUrl(url);
    setStatus('loading');
    stopCamera();
    try {
      const res = await fetch(`${API_BASE}/api/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Backend request failed');
      setResult(data);
      setStatus('result');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  const scanFrame = () => {
    if (status !== 'scanning') return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    if (!video.videoWidth || !video.videoHeight) {
      rafRef.current = requestAnimationFrame(scanFrame);
      return;
    }
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (window.jsQR) {
      const code = window.jsQR(imageData.data, imageData.width, imageData.height);
      if (code && code.data) { processUrl(code.data); return; }
    }
    rafRef.current = requestAnimationFrame(scanFrame);
  };

  const startCameraAndScan = async () => {
    setError('');
    setResult(null);
    setDetectedUrl('');
    setStatus('booting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus('scanning');
    } catch (err) {
      setError(`Camera access failed: ${err.message}`);
      setStatus('error');
    }
  };

  useEffect(() => {
    if (status === 'scanning') rafRef.current = requestAnimationFrame(scanFrame);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [status]);

  useEffect(() => {
    startCameraAndScan();
    return () => stopCamera();
  }, []);

  // Derived result values
  const blendPrediction = result?.blend?.final_prediction ?? 'unknown';
  const blendScore = result?.blend?.final_score ?? 0;
  const blendClass = scoreClass(blendPrediction);
  const blendPct = scorePct(blendScore);

  const logregPrediction = result?.split_a?.logreg?.prediction ?? 'unknown';
  const logregScore = result?.split_a?.logreg?.score ?? 0;

  const bertPrediction = result?.split_b?.distilbert?.prediction ?? 'unknown';
  const bertScore = result?.split_b?.distilbert?.score ?? 0;

  const isPhishing = blendClass === 'is-phishing';
  const verdictIcon = blendClass === 'is-phishing' ? '⚠' : blendClass === 'is-safe' ? '✓' : '?';

  return (
    <>
      <Script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.js" strategy="afterInteractive" />

      <main className="scanPage">
        <video ref={videoRef} playsInline muted className="cameraFeed" />
        <canvas ref={canvasRef} hidden />

        {/* ── Booting ── */}
        {status === 'booting' && (
          <div className="overlay booting">
            <div className="bootRing" />
            <p>Starting camera…</p>
          </div>
        )}

        {/* ── Scanning ── */}
        {status === 'scanning' && (
          <div className="overlay scanning">
            <div className="hudTop">
              <Link href="/" className="backLink">← Back</Link>
              <span>LIVE QR SCAN</span>
            </div>

            <div className="scanFrame" />

            <p className="hint">Point camera at a QR code</p>
          </div>
        )}

        {/* ── Loading ── */}
        {status === 'loading' && (
          <div className="overlay loading">
            <div className="spinner" />
            <h1>Analyzing destination</h1>
            <p className="loadingUrl">{detectedUrl}</p>
            <div className="loadingSteps">
              <div className="loadStep">
                <div className="loadStepDot" />
                Running feature extraction
              </div>
              <div className="loadStep">
                <div className="loadStepDot" />
                Fetching webpage content
              </div>
              <div className="loadStep">
                <div className="loadStepDot" />
                Blending pipeline outputs
              </div>
            </div>
          </div>
        )}

        {/* ── Result ── */}
        {status === 'result' && (
          <div className={`overlay result ${blendClass}`}>
            <h1>Pipeline Result</h1>

            <div className={`verdictCard ${blendClass}`}>
              {/* Header row */}
              <div className="verdictHeader">
                <div>
                  <p className="verdictLabel">Final Verdict</p>
                  <p className={`verdictBadge ${blendClass}`}>
                    {String(blendPrediction).toUpperCase()}
                  </p>
                </div>
                <span className="verdictIcon">{verdictIcon}</span>
              </div>

              {/* URL */}
              <p className="verdictLabel">Scanned URL</p>
              <p className="verdictUrl">{result?.url || detectedUrl}</p>

              {/* Confidence bar */}
              <div className="confRow">
                <p className="verdictLabel">Confidence</p>
                <span className="confPct">{blendPct}</span>
              </div>
              <div className="confBar">
                <div className={`confFill ${blendClass}`} style={{ width: blendPct }} />
              </div>

              {/* Pipeline breakdown */}
              <div className="pipelineBreakdown">
                <div className="pipeStep">
                  <span className="pipeStepLabel">
                    <span className="pipeStepTag">A</span>
                    LogReg · Feature Matrix
                  </span>
                  <span className={`pipeStepScore ${scoreClass(logregPrediction)}`}>
                    {scorePct(logregScore)} · {String(logregPrediction).toUpperCase()}
                  </span>
                </div>
                <div className="pipeStep">
                  <span className="pipeStepLabel">
                    <span className="pipeStepTag">B</span>
                    DistilBERT · Webpage
                  </span>
                  <span className={`pipeStepScore ${scoreClass(bertPrediction)}`}>
                    {scorePct(bertScore)} · {String(bertPrediction).toUpperCase()}
                  </span>
                </div>
                <div className="pipeStep">
                  <span className="pipeStepLabel">
                    <span className="pipeStepTag">⊕</span>
                    XGBoost Blend
                  </span>
                  <span className={`pipeStepScore ${blendClass}`}>
                    {blendPct} · {String(blendPrediction).toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            <div className="actionRow">
              <button onClick={startCameraAndScan}>Scan Again</button>
              <Link href="/" className="ghostBtn">Home</Link>
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {status === 'error' && (
          <div className="overlay error">
            <h1>Scan Failed</h1>
            <p>{error || 'An unknown error occurred.'}</p>
            <div className="actionRow">
              <button onClick={startCameraAndScan}>Retry</button>
              <Link href="/" className="ghostBtn">Home</Link>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
