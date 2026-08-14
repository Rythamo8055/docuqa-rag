import { chromium } from 'playwright';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
await page.waitForTimeout(1200);
await page.screenshot({ path: 'shots/01-initial.png' });

// Upload
await page.setInputFiles('input[type=file]', '/home/rahul/development_walkins/careers company assignment/data/sample.pdf');
await page.waitForSelector('.upload-status--ok', { timeout: 90000 }).catch(()=>{});
await page.waitForTimeout(1500);
await page.screenshot({ path: 'shots/02-uploaded.png' });

// Q1: fresh LLM answer (different question -> cache miss)
await page.fill('textarea', 'Explain supervised learning and give a concrete example from the document.');
await page.click('.composer__send');
await page.waitForTimeout(6000); // thinking state
await page.screenshot({ path: 'shots/03-thinking.png' });
// poll for answer
for (let i = 0; i < 60; i++) {
  await page.waitForTimeout(5000);
  const done = await page.evaluate(() => !!document.querySelector('.msg--assistant'));
  if (done) break;
}
await page.waitForTimeout(1000);
await page.screenshot({ path: 'shots/04-answer.png' });

// Q2: chat history + cache badge
await page.fill('textarea', 'What are the three families of machine learning algorithms?');
await page.click('.composer__send');
for (let i = 0; i < 20; i++) {
  await page.waitForTimeout(3000);
  const done = await page.evaluate(() => {
    const msgs = document.querySelectorAll('.msg--assistant');
    return msgs.length >= 2;
  });
  if (done) break;
}
await page.waitForTimeout(1000);
await page.screenshot({ path: 'shots/05-chat-history.png' });

// Error state: simulate network failure for /query (route abort)
await page.route('**/query', r => r.abort());
await page.fill('textarea', 'This request will fail');
await page.click('.composer__send');
await page.waitForTimeout(4000);
await page.screenshot({ path: 'shots/06-error-state.png' });
await page.unroute('**/query');

// Mobile viewport
await page.setViewportSize({ width: 375, height: 720 });
await page.waitForTimeout(800);
await page.screenshot({ path: 'shots/07-mobile.png' });

console.log('ALL SHOTS DONE');
await browser.close();
