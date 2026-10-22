const fs = require("fs");
const path = require("path");

const standaloneDir = path.resolve(__dirname, "../.next/standalone");
const staticSrc = path.resolve(__dirname, "../.next/static");
const staticDest = path.join(standaloneDir, ".next/static");
const publicSrc = path.resolve(__dirname, "../public");
const publicDest = path.join(standaloneDir, "public");

if (fs.existsSync(standaloneDir)) {
  if (fs.existsSync(staticSrc)) {
    fs.cpSync(staticSrc, staticDest, { recursive: true, force: true });
  }
  if (fs.existsSync(publicSrc)) {
    fs.cpSync(publicSrc, publicDest, { recursive: true, force: true });
  }
}
