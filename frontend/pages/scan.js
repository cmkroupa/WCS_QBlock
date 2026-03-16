import Link from 'next/link';
import Script from 'next/script';
import { useEffect, useRef, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001';

function scoreClass(p) {
  const s = String(p || '').toLowerCase();
  if (s === 'phishing') return 'is-phishing';
  if (s === 'benign' || s === 'safe' || s === 'legitimate') return 'is-safe';
  if (s === 'suspicious') return 'is-suspicious';
  return 'is-unknown';
}

function pct(score) {
  return Math.round(Math.max(0, Math.min(1, Number(score) || 0)) * 100);
}

// SHAP feature → one plain English sentence, or null if not notable
function shapToEnglish(feature, rawValue, impact) {
  const v = Number(rawValue);
  const bad = impact > 0;
  if (Math.abs(impact) < 0.02) return null;

  switch (feature) {
    case 'visible_len':
      if (bad && v < 300)  return `Almost no visible text — only ${Math.round(v)} characters`;
      if (bad && v < 800)  return `Very little visible text (${Math.round(v)} chars)`;
      if (!bad && v > 1000) return `Substantial page content (${Math.round(v)} chars)`;
      return null;
    case 'count_tag__input':
      if (bad && v >= 3) return `${Math.round(v)} input fields collecting data`;
      if (bad && v >= 1) return `Input fields present — collects user data`;
      if (!bad && v === 0) return `No input fields`;
      return null;
    case 'count_tag__form':
      if (bad && v >= 2) return `${Math.round(v)} forms — aggressive data collection`;
      if (bad && v >= 1) return `Form present collecting user data`;
      if (!bad && v === 0) return `No forms present`;
      return null;
    case 'count_external_links':
      if (bad && v === 0) return `Zero external links — keeps visitors trapped on-page`;
      if (bad && v <= 2)  return `Only ${Math.round(v)} external link(s) — unusually isolated`;
      if (!bad && v > 5)  return `${Math.round(v)} outbound links — normal for legitimate sites`;
      return null;
    case 'ratio_external_links':
      if (bad && v < 0.05)  return `${Math.round(v * 100)}% of links leave the page`;
      if (!bad && v > 0.3) return `${Math.round(v * 100)}% of links go to external sites`;
      return null;
    case 'count_tag__a':
      if (bad && v === 0) return `No navigation links at all`;
      if (bad && v > 20)  return `Unusually high number of links (${Math.round(v)})`;
      return null;
    case 'count_tag__iframe':
      if (bad && v >= 1) return `${Math.round(v)} iframe(s) — can load hidden content`;
      return null;
    case 'count_tag__script':
      if (bad && v > 15) return `${Math.round(v)} script tags — far above normal`;
      if (bad && v > 8)  return `Heavy script usage (${Math.round(v)})`;
      if (!bad && v <= 3) return `Very few scripts — simple page`;
      return null;
    case 'num_unique_tags':
      if (bad && v < 6)  return `Only ${Math.round(v)} unique HTML elements — skeleton page`;
      if (!bad && v > 15) return `Rich HTML structure (${Math.round(v)} elements)`;
      return null;
    case 'shannon_entropy':
      if (bad && v > 5.5) return `Highly encoded content — may be obfuscated`;
      return null;
    default:
      return null;
  }
}

const FEATURE_LABELS = {
  count_tag__a:             'Anchor links',
  count_tag__input:         'Input fields',
  count_tag__form:          'Forms',
  count_tag__script:        'Script tags',
  count_tag__iframe:        'Iframes',
  count_tag__img:           'Images',
  count_tag__meta:          'Meta tags',
  count_tag__link:          'Link tags',
  count_tag__button:        'Buttons',
  visible_len:              'Visible text length',
  shannon_entropy:          'Content entropy',
  num_unique_tags:          'Unique HTML elements',
  ratio_external_links:     'External link ratio',
  count_external_links:     'External links',
  count_unique_link_domains:'Unique link domains',
  has_password_field:       'Password field',
  has_login_form:           'Login form',
};

function featureLabel(name) {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  return name.replace(/^count_tag__/, '').replace(/_/g, ' ');
}

export default function ScanPage() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef    = useRef(null);

  const [status, setStatus]       = useState('booting');
  const [detectedUrl, setDetectedUrl] = useState('');
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState('');
  const [showDetails, setShowDetails] = useState(false);

  const stopCamera = () => {
    if (rafRef.current)    { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current)  videoRef.current.srcObject = null;
  };

  const processUrl = async (url) => {
    setDetectedUrl(url);
    setStatus('loading');
    stopCamera();
    try {
      const res  = await fetch(`${API_BASE}/api/scan`, {
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
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) {
      rafRef.current = requestAnimationFrame(scanFrame);
      return;
    }
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (window.jsQR) {
      const code = window.jsQR(imageData.data, imageData.width, imageData.height);
      if (code?.data) { processUrl(code.data); return; }
    }
    rafRef.current = requestAnimationFrame(scanFrame);
  };

  const startCameraAndScan = async () => {
    setError(''); setResult(null); setDetectedUrl('');
    setStatus('booting'); setShowDetails(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } }, audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
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

  useEffect(() => { startCameraAndScan(); return () => stopCamera(); }, []);

  // ── Derived values ──
  const verdict   = result?.blend?.final_prediction ?? 'unknown';
  const score     = result?.blend?.final_score ?? 0;
  const cls       = scoreClass(verdict);
  const confidence = pct(score);

  const xgbPred   = result?.split_a?.prediction ?? 'unknown';
  const xgbScore  = result?.split_a?.score ?? 0;
  const bertPred  = result?.split_b?.prediction ?? 'unknown';
  const bertScore = result?.split_b?.score ?? 0;

  const shapData  = result?.shap ?? null;
  const shapMax   = shapData
    ? Math.max(...shapData.numeric_top.map(f => Math.abs(f.impact)), 0.001)
    : 0.001;

  // Collect plain-English bullets from SHAP
  const bullets = shapData
    ? shapData.numeric_top
        .map(({ feature, raw_value, impact }) => {
          const text = shapToEnglish(feature, raw_value, impact);
          return text ? { key: feature, text, bad: impact > 0 } : null;
        })
        .filter(Boolean)
    : [];

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
            <p className="loadingUrl">{detectedUrl}</p>
            <p className="loadingLabel">Analysing…</p>
          </div>
        )}

        {/* ── Result ── */}
        {status === 'result' && (
          <div className={`overlay result ${cls}`}>
            <div className={`verdictCard ${cls}`}>

              {/* Verdict + confidence */}
              <div className="verdictHeader">
                <div>
                  <p className={`verdictBadge ${cls}`}>
                    {cls === 'is-phishing' ? '⚠ PHISHING' : cls === 'is-safe' ? '✓ SAFE' : cls === 'is-suspicious' ? '? SUSPICIOUS' : 'UNKNOWN'}
                  </p>
                  <p className="verdictUrl">{result?.url || detectedUrl}</p>
                </div>
                <div className="confRing">
                  <svg viewBox="0 0 36 36" className="confRingSvg">
                    <circle cx="18" cy="18" r="15.9" fill="none" strokeWidth="3" className="confRingBg" />
                    <circle cx="18" cy="18" r="15.9" fill="none" strokeWidth="3"
                      className={`confRingFill ${cls}`}
                      strokeDasharray={`${confidence} 100`}
                      strokeLinecap="round"
                      style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
                    />
                  </svg>
                  <span className="confRingPct">{confidence}%</span>
                </div>
              </div>

              {/* Why — plain English signals */}
              {bullets.length > 0 && (
                <div className="signalList">
                  {bullets.map(b => (
                    <div key={b.key} className={`signalItem ${b.bad ? 'sigBad' : 'sigGood'}`}>
                      <span className="signalDot" />
                      <span className="signalText">{b.text}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Technical details toggle */}
              <button className="detailsToggle" onClick={() => setShowDetails(v => !v)}>
                {showDetails ? '▲ Hide details' : '▼ Technical details'}
              </button>

              {showDetails && (
                <div className="detailsPanel">

                  {/* Model scores */}
                  <p className="detailsLabel">Model scores</p>
                  <div className="modelRows">
                    {[
                      { tag: 'A', name: 'HTML features', pred: xgbPred,  s: xgbScore  },
                      { tag: 'B', name: 'Embeddings',    pred: bertPred, s: bertScore },
                      { tag: '⊕', name: 'Final blend',   pred: verdict,  s: score     },
                    ].map(row => (
                      <div key={row.tag} className="modelRow">
                        <span className="modelTag">{row.tag}</span>
                        <span className="modelName">{row.name}</span>
                        <span className={`modelScore ${scoreClass(row.pred)}`}>
                          {pct(row.s)}% · {String(row.pred).toUpperCase()}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* SHAP bars */}
                  {shapData && (
                    <>
                      <p className="detailsLabel" style={{ marginTop: 14 }}>Feature weights (SHAP)</p>
                      <div className="shapBars">
                        {shapData.numeric_top.map(f => (
                          <div key={f.feature} className="shapRow">
                            <span className="shapName">{featureLabel(f.feature)}</span>
                            <div className="shapTrack">
                              <div
                                className={`shapFill ${f.impact >= 0 ? 'shap-red' : 'shap-green'}`}
                                style={{ width: `${Math.round((Math.abs(f.impact) / shapMax) * 100)}%` }}
                              />
                            </div>
                            <span className={`shapVal ${f.impact >= 0 ? 'shap-red' : 'shap-green'}`}>
                              {f.impact >= 0 ? '+' : ''}{f.impact.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>
                      <p className="shapLegend">
                        <span className="shap-red">red = toward phishing</span>
                        &nbsp;·&nbsp;
                        <span className="shap-green">green = toward safe</span>
                      </p>
                    </>
                  )}
                </div>
              )}
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
            <p className="errorTitle">Scan failed</p>
            <p className="errorMsg">{error || 'An unknown error occurred.'}</p>
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
