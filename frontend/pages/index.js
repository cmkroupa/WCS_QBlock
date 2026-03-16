import Link from 'next/link';

export default function HomePage() {
  return (
    <>
      <main className="home">

        {/* ── Hero ── */}
        <section className="homeHero">
          <h1>Scan before you visit</h1>
          <p className="homeSub">
            QBlock authenticates it for you.
          </p>
          <Link href="/scan" className="homeCta">Scan</Link>
        </section>

        {/* ── Three columns ── */}
        <div className="homeGrid">
          <div className="homeGridItem">
            <span className="homeGridN">01</span>
            <strong>The problem</strong>
            <p>QR codes encode a URL invisibly. Unlike a link you can hover over, there is no way to inspect the destination before your browser loads it — making them a reliable delivery mechanism for phishing attacks.</p>
          </div>
          <div className="homeGridItem">
            <span className="homeGridN">02</span>
            <strong>The fix</strong>
            <p>QBlock intercepts the URL from the QR code and checks it before anything opens. It fetches the page, analyses it, and returns a verdict — safe, suspicious, or phishing — in a few seconds.</p>
          </div>
          <div className="homeGridItem">
            <span className="homeGridN">03</span>
            <strong>The pipeline</strong>
            <p>One model inspects the raw HTML structure — forms, links, scripts, entropy. A second reads the page content using a transformer. A meta model weighs both and produces the final score.</p>
          </div>
        </div>

      </main>
    </>
  );
}
