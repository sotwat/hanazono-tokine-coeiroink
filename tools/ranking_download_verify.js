// Verify completed browser downloads against the independent model inventory.
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const [root, inventory, series, epochArg, downloads = '/Users/aaa/Downloads'] = process.argv.slice(2);
const epoch = Number(epochArg);
if (!root || !inventory || !['control', 'quarter_lr'].includes(series) || !Number.isInteger(epoch)) {
  throw Error('Usage: node ranking_download_verify.js ROOT INVENTORY SERIES EPOCH [DOWNLOADS]');
}
const item = JSON.parse(fs.readFileSync(inventory)).rows.find(row => row.series === series && row.epoch === epoch);
if (!item) throw Error('Candidate missing from inventory');
const destination = path.join(root, 'ranking_models', series, `${epoch}epoch.pth`);
function digest(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}
let watcher, timer;
function finish(source) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  if (source !== destination) fs.copyFileSync(source, destination, fs.constants.COPYFILE_EXCL);
  if (digest(destination) !== item.sha256) throw Error('Copied model SHA mismatch');
  if (watcher) watcher.close();
  if (timer) clearTimeout(timer);
  process.stdout.write(JSON.stringify({ series, epoch, size: item.size, sha256: item.sha256, verified: true }) + '\n');
  process.exit(0);
}
function check() {
  if (fs.existsSync(destination)) {
    if (fs.statSync(destination).size !== item.size || digest(destination) !== item.sha256) {
      throw Error('Existing destination does not match inventory; preserve and inspect it');
    }
    return finish(destination);
  }
  for (const name of fs.readdirSync(downloads)) {
    if (!name.startsWith(`${epoch}epoch`) || !name.endsWith('.pth')) continue;
    const file = path.join(downloads, name);
    if (fs.statSync(file).size === item.size && digest(file) === item.sha256) return finish(file);
  }
}
watcher = fs.watch(downloads, (_event, name) => {
  if (name && name.startsWith(`${epoch}epoch`) && name.endsWith('.pth')) check();
});
timer = setTimeout(() => {
  watcher.close();
  process.stderr.write('Model download not complete within 50 seconds; inspect browser state before retrying.\n');
  process.exit(2);
}, 50000);
check();
