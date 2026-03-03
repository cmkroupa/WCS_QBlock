import Link from 'next/link';

export default function HomePage() {
  return (
    <>
      <nav className="siteNav">
        <Link href="/" className="navBrand">
          <span className="navBrandMark">Q</span>
          CODEX QBLOCK
        </Link>
        <Link href="/scan" className="navScanBtn">Scan QR</Link>
      </nav>

      <main className="landingPage">
        <div className="landingGlow glowA" />
        <div className="landingGlow glowB" />

        {/* ── Hero ── */}
        <section className="hero">
          <p className="kicker">QR URL PHISHING BLOCKER</p>
          <h1>Stop phishing<br />before it starts.</h1>
          <p className="heroText">
            QR codes can conceal malicious destinations until the moment a user scans.
            This project intercepts that moment — extracts the URL, engineers 35 features,
            runs a dual-branch ML pipeline, and returns a verdict before users ever trust
            the destination.
          </p>
          <div className="heroActions">
            <Link href="/scan" className="scanBtn">Scan a QR Code</Link>
            <a href="#how" className="ghostLink">See the Pipeline</a>
          </div>
        </section>

        {/* ── What ── */}
        <section id="what" className="storySection">
          <div className="sectionTag">What This Is</div>
          <h2>A real-time QR safety gateway.</h2>
          <p>
            The frontend is a live camera scanner built in Next.js. The backend is a Python
            service that receives scanned URLs, engineers a 35-feature extraction matrix,
            fetches page content, and executes the split pipeline before returning a
            phishing classification with per-branch scores.
          </p>
        </section>

        {/* ── Why ── */}
        <section id="why" className="storySection accent">
          <div className="sectionTag">Why It Matters</div>
          <h2>Phishing moved to physical surfaces.</h2>
          <p>
            A poster, restaurant menu, or parking sticker QR can redirect victims to
            credential-harvesting pages. Traditional link-awareness training fails
            because the destination is never visible before scanning. This workflow
            adds automated inspection at scan-time — closing that gap entirely.
          </p>
        </section>

        {/* ── How ── */}
        <section id="how" className="storySection">
          <div className="sectionTag">How We Do It</div>
          <h2>Two branches, one blended decision.</h2>
          <div className="howGrid">
            <article>
              <span className="howNum">01</span>
              <h3>Feature Extraction</h3>
              <p>
                35 URL-derived signals — TLD risk, brand Levenshtein distance,
                subdomain depth, entropy, path structure, and more — fed into
                the Logistic Regression stage.
              </p>
            </article>
            <article>
              <span className="howNum">02</span>
              <h3>DistilBERT Analysis</h3>
              <p>
                The raw URL and live-fetched webpage content are passed to a
                DistilBERT model that evaluates semantic phishing signals
                feature engineering alone cannot capture.
              </p>
            </article>
            <article>
              <span className="howNum">03</span>
              <h3>XGBoost Blend</h3>
              <p>
                Both branch scores are combined by an XGBoost blending stage
                that weights each branch's confidence, producing a single
                calibrated phishing probability.
              </p>
            </article>
          </div>
        </section>

        {/* ── Pipeline visualization ── */}
        <section className="pipelineSection">
          <div className="sectionTag">Pipeline Architecture</div>
          <h2 style={{ margin: '0 0 4px', fontSize: 'clamp(1.1rem, 2vw, 1.55rem)' }}>
            Scan to verdict in one request.
          </h2>
          <p style={{ margin: '8px 0 0', color: 'var(--ink-soft)', fontSize: '0.9rem', lineHeight: 1.6 }}>
            Every QR scan triggers the full pipeline synchronously and returns structured
            per-branch results alongside the blended final decision.
          </p>

          <div className="pipeFlow">
            <div className="pipeBlock">
              <div className="pipeBlockNum">01</div>
              <div className="pipeBlockTitle">QR → URL</div>
              <div className="pipeBlockDesc">
                Camera frame decode, URL extraction, protocol normalisation
              </div>
            </div>

            <div className="pipeFlowArrow">→</div>

            <div className="pipeBlock dual">
              <div className="pipeBlockNum">02</div>
              <div className="pipeBlockTitle">Split()</div>
              <div className="pipeBlockPair">
                <div className="pipeHalf">
                  <span className="halfTag">A</span>
                  Feature matrix · Logistic Regression
                </div>
                <div className="pipeHalf">
                  <span className="halfTag">B</span>
                  DistilBERT · URL + Webpage
                </div>
              </div>
            </div>

            <div className="pipeFlowArrow">→</div>

            <div className="pipeBlock">
              <div className="pipeBlockNum">03</div>
              <div className="pipeBlockTitle">Blend()</div>
              <div className="pipeBlockDesc">
                XGBoost combines branch scores into one calibrated output
              </div>
            </div>

            <div className="pipeFlowArrow">→</div>

            <div className="pipeBlock verdict">
              <div className="pipeBlockNum">04</div>
              <div className="pipeBlockTitle">Verdict</div>
              <div className="pipeBlockDesc">
                Safe or Phishing returned to scanner UI with per-branch detail
              </div>
            </div>
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="finalCta">
          <h2>Ready to test a live QR?</h2>
          <Link href="/scan" className="scanBtn">Open Full Scanner</Link>
        </section>
      </main>
    </>
  );
}
