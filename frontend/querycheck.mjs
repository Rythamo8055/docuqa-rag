import { chromium } from 'playwright';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);
await page.setInputFiles('input[type=file]', '/home/rahul/development_walkins/careers company assignment/data/sample.pdf');
await page.waitForSelector('.upload-status--ok', { timeout: 90000 });
await page.waitForTimeout(2000);

await page.fill('textarea', 'What are the three families of machine learning algorithms?');
await page.click('.composer__send');
await page.waitForTimeout(25000);

const result = await page.evaluate(() => ({
  messages: [...document.querySelectorAll('.msg')].map(m => ({
    role: m.className.includes('user') ? 'user' : 'assistant',
    text: m.querySelector('.msg__text')?.textContent?.slice(0, 250),
  })),
  badges: [...document.querySelectorAll('.badge')].map(b => b.textContent?.trim()),
  citations: [...document.querySelectorAll('.citation')].map(c => ({
    head: c.querySelector('.citation__head')?.textContent?.trim(),
    text: c.querySelector('.citation__text')?.textContent?.slice(0, 80),
  })),
  meta: [...document.querySelectorAll('.msg__meta')].map(m => m.textContent?.trim()),
  thinking: !!document.querySelector('.thinking'),
}));
console.log(JSON.stringify(result, null, 1));
await page.screenshot({ path: '04-answer.png' });
await browser.close();
