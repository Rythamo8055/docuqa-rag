import { chromium } from 'playwright';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);
await page.setInputFiles('input[type=file]', '/home/rahul/development_walkins/careers company assignment/data/sample.pdf');
await page.waitForSelector('.upload-status--ok', { timeout: 90000 }).catch(() => console.log('upload slow'));
await page.waitForTimeout(2000);

await page.fill('textarea', 'What are the three families of machine learning algorithms?');
await page.click('.composer__send');
// poll until assistant message or error appears (up to 150s)
let state = null;
for (let i = 0; i < 30; i++) {
  await page.waitForTimeout(5000);
  state = await page.evaluate(() => ({
    hasAssistant: !!document.querySelector('.msg--assistant'),
    thinking: !!document.querySelector('.thinking'),
    alerts: [...document.querySelectorAll('.alert')].map(a => a.textContent?.trim().slice(0, 150)),
  }));
  if (state.hasAssistant || state.alerts.length) break;
}
const result = await page.evaluate(() => ({
  messages: [...document.querySelectorAll('.msg')].map(m => ({
    role: m.className.includes('user') ? 'user' : 'assistant',
    text: m.querySelector('.msg__text')?.textContent?.slice(0, 300),
  })),
  badges: [...document.querySelectorAll('.badge')].map(b => b.textContent?.trim()),
  citations: document.querySelectorAll('.citation').length,
  meta: [...document.querySelectorAll('.msg__meta')].map(m => m.textContent?.trim()),
}));
console.log(JSON.stringify({ poll: state, result }, null, 1));
await page.screenshot({ path: '04-answer.png' });
await browser.close();
