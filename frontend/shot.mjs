import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const logs = [];
page.on('console', m => logs.push(m.type() + ': ' + m.text()));

// 1) Initial state (backend online, no doc indexed)
await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);
await page.screenshot({ path: '01-initial.png' });
console.log('01 done');

// 2) Upload the sample PDF
await page.setInputFiles('input[type=file]', '/home/rahul/development_walkins/careers company assignment/data/sample.pdf');
await page.waitForSelector('.upload-status--ok', { timeout: 60000 }).catch(() => console.log('upload status not found'));
await page.waitForTimeout(4000);
await page.screenshot({ path: '02-uploaded.png' });
console.log('02 done');

// 3) Ask a question
await page.fill('textarea', 'What are the three families of machine learning algorithms?');
await page.screenshot({ path: '03-question-typed.png' });
await page.click('.composer__send');
await page.waitForTimeout(15000); // LLM generation
await page.screenshot({ path: '04-answer.png' });
console.log('04 done');

// 4) Chat history + citations state
await page.fill('textarea', 'What does RAG stand for and why is it useful?');
await page.click('.composer__send');
await page.waitForTimeout(15000);
await page.screenshot({ path: '05-chat-history.png' });
console.log('05 done');

console.log('CONSOLE ERRORS:', logs.filter(l => l.startsWith('error')).slice(0, 5));
await browser.close();
