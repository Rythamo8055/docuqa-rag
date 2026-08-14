import { chromium } from 'playwright';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

// After upload state - check DOM
const initial = await page.evaluate(() => ({
  hasUploadPanel: !!document.querySelector('.dropzone'),
  hasChatTextarea: !!document.querySelector('textarea'),
  emptyTitle: document.querySelector('.empty__title')?.textContent,
  statusDots: [...document.querySelectorAll('.dot')].map(d => d.className),
}));
console.log('INITIAL:', JSON.stringify(initial, null, 1));

await page.setInputFiles('input[type=file]', '/home/rahul/development_walkins/careers company assignment/data/sample.pdf');
await page.waitForTimeout(30000); // wait for ingest (models load)

const after = await page.evaluate(() => ({
  hasChatTextarea: !!document.querySelector('textarea'),
  emptyTitle: document.querySelector('.empty__title')?.textContent,
  uploadStatus: document.querySelector('.upload-status')?.textContent?.trim(),
  alerts: [...document.querySelectorAll('.alert')].map(a => a.textContent?.trim().slice(0, 120)),
  statValues: [...document.querySelectorAll('.stat__value')].map(v => v.textContent),
}));
console.log('AFTER UPLOAD:', JSON.stringify(after, null, 1));
await browser.close();
