const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');


const API_KEY = 'gsk_7YlLERcc5HCZb6wZMLfJWGdyb3FYS3l6TalS3CRB0mp2QtjIAG1c1chsuya';

http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  if (req.method === 'GET') {
    const filePath = path.join(__dirname, 'index.html');
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); res.end('Not found'); return; }
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(data);
    });
    return;
  }

  if (req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      let parsed;
      try { parsed = JSON.parse(body); } catch(e) {
        res.writeHead(400); res.end(JSON.stringify({ error: { message: 'Invalid JSON' } })); return;
      }

      const groqBody = JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        max_tokens: 4096,
        temperature: 0.7,
        messages: parsed.messages
      });

      const options = {
        hostname: 'api.groq.com',
        path: '/openai/v1/chat/completions',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_KEY}`,
          'Content-Length': Buffer.byteLength(groqBody)
        }
      };

      const apiReq = https.request(options, apiRes => {
        let data = '';
        apiRes.on('data', chunk => data += chunk);
        apiRes.on('end', () => {
          try {
            const groqRes = JSON.parse(data);
            if (groqRes.error) {
              res.writeHead(400);
              res.end(JSON.stringify({ error: { message: groqRes.error.message } }));
              return;
            }
            const text = groqRes.choices[0].message.content;
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ content: [{ type: 'text', text }] }));
          } catch(e) {
            res.writeHead(500);
            res.end(JSON.stringify({ error: { message: 'Parse error: ' + e.message } }));
          }
        });
      });

      apiReq.on('error', e => {
        res.writeHead(500);
        res.end(JSON.stringify({ error: { message: e.message } }));
      });

      apiReq.write(groqBody);
      apiReq.end();
    });
    return;
  }

  res.writeHead(405); res.end();

}).listen(3000, () => {
  console.log('✅ AI Resume Builder running at http://localhost:3000');
});
