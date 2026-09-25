import fs from 'fs';
import path from 'path';

function walkDir(dir, callback) {
  fs.readdirSync(dir).forEach(f => {
    let dirPath = path.join(dir, f);
    let isDirectory = fs.statSync(dirPath).isDirectory();
    if (isDirectory) {
      walkDir(dirPath, callback);
    } else {
      callback(dirPath);
    }
  });
}

const notesDir = path.resolve('notes');
let totalSubscriptsFixed = 0;
let totalDisplayMathFixed = 0;
let filesModified = 0;

walkDir(notesDir, (filePath) => {
  if (!filePath.endsWith('.md')) return;

  const content = fs.readFileSync(filePath, 'utf8');
  let newContent = content;

  // 1. Sanitize escaped underscores inside math and LaTeX expressions
  newContent = newContent.replace(/\\_([a-zA-Z0-9{])/g, '_$1').replace(/\\_/g, '_');

  // 2. Format standalone single-line display math $$ formula $$ into multi-line $$ blocks
  // e.g. "   $$ formula $$" -> "   $$\n   formula\n   $$"
  newContent = newContent.replace(/^([ \t]*)\$\$\s*([^\n$]+?)\s*\$\$\s*$/gm, (match, indent, formula) => {
    totalDisplayMathFixed++;
    return `${indent}$$\n${indent}${formula}\n${indent}$$`;
  });

  if (newContent !== content) {
    fs.writeFileSync(filePath, newContent, 'utf8');
    filesModified++;
    console.log(`Updated math in ${path.relative(process.cwd(), filePath)}`);
  }
});

console.log(`\nDone! Formatted ${totalDisplayMathFixed} display math blocks across ${filesModified} files.`);

