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


const FEATURE_LABELS = {
  count_tag__a: 'a tags',
  count_tag__input: 'input tags',
  count_tag__form: 'form tags',
  count_tag__script: 'script tags',
  count_tag__iframe: 'iframe tags',
  count_tag__img: 'images',
  count_tag__meta: 'meta tags',
  count_tag__link: 'link tags',
  count_tag__button: 'buttons',
  count_tag__svg: 'svg tags',
  visible_len: 'visible text len',
  shannon_entropy: 'shannon entropy',
  num_unique_tags: 'num unique elements',
  count_external_links: 'external links',
  count_internal_links: 'internal links',
  max_dom_depth: 'max DOM depth',
  raw_len: 'total HTML size',
};

function featureLabel(name) {
  return FEATURE_LABELS[name] || name.replace(/^count_tag__/, '').replace(/_/g, ' ');
}


function parseUrl(urlStr) {
  try {
    const u = new URL(urlStr.startsWith('http') ? urlStr : 'https://' + urlStr);
    return {
      protocol: u.protocol.replace(':', ''),
      host: u.hostname,
      path: u.pathname !== '/' ? u.pathname : '',
      query: u.search,
    };
  } catch {
    return { host: urlStr };
  }
}


function ScoreRing({ score }) {
  const pred = score >= 0.5 ? 'phishing' : score >= 0.25 ? 'suspicious' : 'safe';
  const cls = scoreClass(pred);
  const p = pct(score);
  return (
    <div className="plScoreRing">
      <svg viewBox="0 0 36 36" className="plScoreRingSvg">
        <circle cx="18" cy="18" r="15.9" fill="none" strokeWidth="3" className="confRingBg" />
        <circle
          cx="18" cy="18" r="15.9" fill="none" strokeWidth="3"
          className={`confRingFill ${cls}`}
          strokeDasharray={`${p} 100`}
          strokeLinecap="round"
          style={{ transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }}
        />
      </svg>
      <span className={`plScoreRingPct ${cls}`}>{p}%</span>
    </div>
  );
}

function PlStage({ idx, icon, title, explain, badge, badgeCls, visible, isLast, children }) {
  return (
    <div className={`plStage ${visible ? 'plStageVisible' : ''}`}>
      <div className="plStageLeft">
        <div className={`plStageNode ${visible ? 'plStageNodeActive' : ''}`}>
          <span className="plStageNum">{String(idx).padStart(2, '0')}</span>
        </div>
        {!isLast && <div className={`plStageLine ${visible ? 'plStageLineActive' : ''}`} />}
      </div>
      <div className="plStageCard">
        <div className="plStageCardHeader">
          <span className="plStageTitle">{title}</span>
          {badge && <span className={`plStageBadge ${badgeCls || ''}`}>{badge}</span>}
        </div>
        <div className="plStageCardBody">{children}</div>
      </div>
    </div>
  );
}

export default function PipelinePage() {
  const [inputUrl, setInputUrl] = useState('');
  const [analysedUrl, setAnalysedUrl] = useState('');
  const [status, setStatus] = useState('idle');
  const [result, setResult] = useState(null);
  const [visibleCount, setVisibleCount] = useState(0);
  const [textTab, setTextTab] = useState('visible');
  const [error, setError] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (status !== 'result' || !result) return;
    setVisibleCount(0);
    let count = 0;
    const id = setInterval(() => {
      count++;
      setVisibleCount(count);
      if (count >= 8) clearInterval(id);
    }, 320);
    return () => clearInterval(id);
  }, [status]);

  const stopCamera = () => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  useEffect(() => () => stopCamera(), []);

  const scanFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) {
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
      if (code?.data) { stopCamera(); setStatus('idle'); analyse(code.data); return; }
    }
    rafRef.current = requestAnimationFrame(scanFrame);
  };

  const startQr = async () => {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } }, audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus('qr');
      rafRef.current = requestAnimationFrame(scanFrame);
    } catch (err) {
      setError(`Camera: ${err.message}`);
    }
  };

  const analyse = async (targetUrl) => {
    const u = (targetUrl || inputUrl).trim();
    if (!u) return;
    setAnalysedUrl(u);
    setStatus('loading');
    setVisibleCount(0);
    setResult(null);
    setError('');
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90_000);
    try {
      const res = await fetch(`${API_BASE}/api/pipeline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: u }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Backend request failed');
      setResult(data);
      setStatus('result');
    } catch (err) {
      clearTimeout(timeoutId);
      setError(err.name === 'AbortError' ? 'Request timed out — backend took over 90s' : err.message);
      setStatus('error');
    }
  };

  const reset = () => {
    setStatus('idle'); setResult(null); setVisibleCount(0);
    setError(''); setAnalysedUrl(''); setTextTab('visible');
  };

  const s = result?.stages;

  return (
    <>
      <Script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.js" strategy="afterInteractive" />

      <video
        ref={videoRef} playsInline muted
        className={status === 'qr' ? 'plQrVideoActive' : 'plQrVideoHidden'}
      />
      <canvas ref={canvasRef} hidden />

      {status === 'qr' && (
        <div className="plQrOverlay">
          <button className="plQrClose" onClick={() => { stopCamera(); setStatus('idle'); }}>
            ✕ Close
          </button>
        </div>
      )}

      <div className="plPage">
        {/* ── Nav ── */}
        <nav className="siteNav">
          <Link href="/" className="navBrand">
            QBlock
          </Link>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <Link href="/scan" style={{ fontSize: '0.85rem', color: 'var(--ink-soft)', textDecoration: 'none', padding: '7px 12px' }}>
              Scan
            </Link>
            <Link href="/pipeline" className="navScanBtn">Pipeline</Link>
          </div>
        </nav>

        {/* ── Hero ── */}
        <section className="plHero">
          <h1 className="plHeroTitle">QBlock Pipeline</h1>
          <div className="plInputRow">
            <input
              className="plInput"
              type="text"
              placeholder="https://example.com"
              value={inputUrl}
              onChange={e => setInputUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && status !== 'loading' && analyse()}
              disabled={status === 'loading'}
              spellCheck={false}
            />
            <button
              className="plAnalyseBtn"
              onClick={() => analyse()}
              disabled={status === 'loading' || !inputUrl.trim()}
            >
              {status === 'loading' ? <span className="plBtnSpinner" /> : 'Analyse'}
            </button>
            <button
              className="plQrBtn"
              onClick={startQr}
              disabled={status === 'loading'}
              title="Scan QR code"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="7" rx="1.5" />
                <rect x="14" y="3" width="7" height="7" rx="1.5" />
                <rect x="3" y="14" width="7" height="7" rx="1.5" />
                <path d="M14 14h3v3h-3zM19 14v2M14 19h2M19 19h2v2h-2z" />
              </svg>
            </button>
          </div>
          {error && status !== 'error' && <p className="plErrorInline">{error}</p>}
        </section>

        {/* ── Loading ── */}
        {status === 'loading' && (
          <div className="plLoading">
            <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
            <div>
              <p className="plLoadingUrl">{analysedUrl}</p>
              <p className="plLoadingInfo">
                Fetching page · extracting features · embedding · scoring…
              </p>
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {status === 'error' && (
          <div className="plErrorCard">
            <p className="errorTitle" style={{ fontSize: '1rem' }}>Analysis failed</p>
            <p className="errorMsg">{error}</p>
            <button onClick={reset}>Try again</button>
          </div>
        )}

        {/* ── Pipeline stages ── */}
        {status === 'result' && s && (
          <div className="plPipeline">

            {/* ── 1: URL Analysis ── */}
            {(() => {
              const { score, signals, original_url, final_url, redirected } = s.url_analysis;
              const resolvedUrl = final_url || result.url;
              const parts = parseUrl(resolvedUrl);
              const pred = score >= 0.5 ? 'phishing' : score >= 0.25 ? 'suspicious' : 'safe';
              const cls = scoreClass(pred);
              return (
                <PlStage idx={1} title="URL Analysis"
                  badgeCls={cls}
                  visible={visibleCount >= 1} isLast={false}>
                  {redirected && (
                    <div className="plRedirectRow">
                      <span className="plRedirectOriginal">{original_url}</span>
                      <span className="plRedirectArrow">→</span>
                      <span className="plRedirectFinal">{final_url}</span>
                    </div>
                  )}
                  <div className="plUrlBreak">
                    {parts.protocol && <span className="plUrlChip plUrlProtocol">{parts.protocol}</span>}
                    <span className="plUrlChipSep">://</span>
                    <span className="plUrlChip plUrlHost">{parts.host}</span>
                    {parts.path && <span className="plUrlChip plUrlPath">{parts.path}</span>}
                    {parts.query && <span className="plUrlChip plUrlQuery">{parts.query}</span>}
                  </div>
                </PlStage>
              );
            })()}

            {/* ── 2: Fetch HTML ── */}
            {(() => {
              const { html_bytes, title } = s.fetch;
              const kb = (html_bytes / 1024).toFixed(1);
              return (
                <PlStage idx={2} title="Fetch HTML"
                  visible={visibleCount >= 2} isLast={false}>

                  <div className="plFetchStat">
                    <span className="plStatLabel">Method</span>
                    <span className="plFetchVal">Playwright </span>
                  </div>
                </PlStage>
              );
            })()}

            {/* ── 3: Feature Extraction ── */}
            {(() => {
              const entries = Object.entries(s.html_features.features);
              return (
                <PlStage idx={3} title="Feature Extraction"
                  visible={visibleCount >= 3} isLast={false}>
                  <div className="plFeatureGrid">
                    {entries.map(([name, value]) => {
                      const disp = typeof value === 'number' && !Number.isInteger(value)
                        ? value.toFixed(3) : String(value);
                      return (
                        <div key={name} className="plFeatureCell">
                          <span className="plFcName">{featureLabel(name)}</span>
                          <span className="plFcValue">{disp}</span>
                        </div>
                      );
                    })}
                  </div>
                </PlStage>
              );
            })()}

            {/* ── 4: Text Extraction ── */}
            {(() => {
              const { visible_text, visible_text_len, structural_core, structural_core_len } = s.text_extraction;
              return (
                <PlStage idx={4} title="Text Extraction"
                  visible={visibleCount >= 4} isLast={false}>
                  <div className="plChannelRow">
                    <div className="plChannel">
                      <span className="plChannelLabel">Visible text</span>
                      <span className="plChannelDesc">Words the page shows the user</span>
                    </div>
                    <div className="plChannel">
                      <span className="plChannelLabel">Structural core</span>
                      <span className="plChannelDesc">Links, forms, and page actions</span>
                    </div>
                  </div>
                  <div className="plTabs">
                    <button className={`plTab ${textTab === 'visible' ? 'plTabActive' : ''}`}
                      onClick={() => setTextTab('visible')}>
                      Visible text <span className="plTabCount">{visible_text_len.toLocaleString()} chars</span>
                    </button>
                    <button className={`plTab ${textTab === 'struct' ? 'plTabActive' : ''}`}
                      onClick={() => setTextTab('struct')}>
                      Structural core <span className="plTabCount">{structural_core_len.toLocaleString()} chars</span>
                    </button>
                  </div>
                  <div className="plTextBox">
                    {(textTab === 'visible' ? visible_text : structural_core) || <em className="plMuted">Empty</em>}
                  </div>
                </PlStage>
              );
            })()}

            {/* ── 5: Embedding ── */}
            {(() => {
              const { voter_b_model, bert_input_len } = s.embedding;
              return (
                <PlStage idx={5} title="Transformer Embedding"
                  visible={visibleCount >= 5} isLast={false}>
                  <div className="plEmbedFlow">
                    <div className="plEmbedBlock">Visible</div>
                    <span className="plEmbedArrow">+</span>
                    <div className="plEmbedBlock">Structural</div>
                    <span className="plEmbedArrow">→ combine →</span>
                    <div className="plEmbedBlock">{bert_input_len} chars</div>
                    <span className="plEmbedArrow">→ tokenise →</span>
                    <div className="plEmbedBlock">[CLS] token</div>
                  </div>
                  <p className="plMuted" style={{ marginTop: 8, fontSize: '0.8rem' }}>{voter_b_model}</p>
                </PlStage>
              );
            })()}

            {/* ── 6: Voter A ── */}
            {(() => {
              const { score, prediction, shap_top } = s.voter_a;
              const cls = scoreClass(prediction);
              const shapMax = shap_top.length
                ? Math.max(...shap_top.map(f => Math.abs(f.impact)), 0.001)
                : 0.001;
              return (
                <PlStage idx={6} title="Voter A — XGBoost"
                  badge={`${pct(score)}%`} badgeCls={cls}
                  visible={visibleCount >= 6} isLast={false}>
                  <div className="plVoterRow">
                    <ScoreRing score={score} />
                    <p className="plVoterName">Trained on the HTML features</p>
                  </div>
                  {shap_top.length > 0 && (
                    <div className="shapSection">
                      <p className="shapSectionLabel">Feature contributions</p>
                      <div className="shapBars">
                        {shap_top.map(f => (
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
                    </div>
                  )}
                </PlStage>
              );
            })()}

            {/* ── 7: Voter B ── */}
            {(() => {
              const { score, prediction } = s.voter_b;
              const cls = scoreClass(prediction);
              return (
                <PlStage idx={7} title="Voter B — PhishBERT"
                  badge={`${pct(score)}%`} badgeCls={cls}
                  visible={visibleCount >= 7} isLast={false}>
                  <div className="plVoterRow">
                    <ScoreRing score={score} />
                    <p className="plVoterName">Fine-tuned XLM-RoBERTa · scores page text end-to-end</p>
                  </div>
                </PlStage>
              );
            })()}

            {/* ── 8: Meta Learner ── */}
            {(() => {
              const { inputs, calibrated_score, prediction } = s.meta;
              const finalCls = scoreClass(prediction);
              const voterACls = scoreClass(s.voter_a.prediction);
              const voterBCls = scoreClass(s.voter_b.prediction);
              return (
                <PlStage idx={8} title="Blender"
                  badgeCls={finalCls}
                  visible={visibleCount >= 8} isLast={true}>
                  <div className="plMetaInputs">
                    <div className={`plMetaVoterCard ${voterACls}`}>
                      <span className="plMetaVoterCardLabel">Voter A - XGBoost</span>
                      <span className="plMetaVoterCardScore">{pct(inputs[0])}%</span>
                    </div>
                    <div className={`plMetaVoterCard ${voterBCls}`}>
                      <span className="plMetaVoterCardLabel">Voter B - LogReg on BERT</span>
                      <span className="plMetaVoterCardScore">{pct(inputs[1])}%</span>
                    </div>
                  </div>
                  <div className={`plFinalVerdict ${finalCls}`}>
                    <span className="plFinalBadge">
                      {prediction === 'phishing' ? 'PHISHING'
                        : prediction === 'suspicious' ? 'SUSPICIOUS'
                          : 'SAFE'} — {pct(calibrated_score)}%
                    </span>
                  </div>
                </PlStage>
              );
            })()}

          </div>
        )}
      </div>
    </>
  );
}
