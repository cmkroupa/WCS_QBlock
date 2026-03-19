import Link from 'next/link';

export default function HomePage() {
  return (
    <>
      <main className="home">

        {/* ── Hero ── */}
        <section className="homeHero">
          <h1>QBlock</h1>
          <p className="homeSub">
            Scan before you visit
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
            <Link href="/scan" className="homeCta">Scan</Link>
            <Link href="/pipeline" className="homeCta" style={{ background: 'transparent', border: '1px solid rgba(34,211,238,0.35)', color: 'var(--accent)' }}>Pipeline</Link>
          </div>
        </section>

        {/* ── Three columns ── */}
        <div className="homeGrid">
          <div className="homeGridItem">
            <strong>The problem</strong>
            <p>QR codes are wrappers for URL's. Unlike a link you can hover over, there is no way to inspect the destination before your browser loads it, making them a reliable attack vector for phishing attacks.</p>
          </div>
          <div className="homeGridItem">
            <strong>The fix</strong>
            <p>QBlock extracts the URL from the QR code and checks it before anything opens. It fetches the page, analyzes it, and returns a verdict if it is phishing or not</p>
          </div>
          <div className="homeGridItem">
            <strong>The pipeline</strong>
            <p>One model inspects the raw HTML structure — forms, links, scripts, entropy. A second reads the page content using a transformer. A meta model weighs both and produces the final score.</p>
          </div>
        </div>

      </main>
    </>
  );
}
