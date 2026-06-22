const fs = require('fs');
const path = require('path');

function findQuizFiles(dir, fileList = []) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);
        if (stat.isDirectory()) {
            findQuizFiles(filePath, fileList);
        } else if (file.endsWith('.md') && (filePath.includes('quiz') || filePath.includes('quizzes') || file.includes('quiz'))) {
            fileList.push(filePath);
        }
    }
    return fileList;
}

const rootDir = path.resolve(__dirname, '..');
const quizFiles = findQuizFiles(rootDir);
console.log(`Found ${quizFiles.length} quiz files:`, quizFiles);

quizFiles.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Check if the file needs a blank line before option A
    // We match any line that does not start with #, -, *, |, or space, and is not empty,
    // followed immediately by option A.
    // Options pattern: A) or A. or A] or A 
    const originalContent = content;
    
    // We want to insert an empty line if there is a line with text followed immediately by a line starting with A) or A.
    // Example:
    // Which Kubernetes component is responsible for scheduling Pods to nodes?
    // A) kubelet
    // Should become:
    // Which Kubernetes component is responsible for scheduling Pods to nodes?
    // 
    // A) kubelet
    // Note: We avoid matching headers, lists, blockquotes, code blocks, etc.
    // So the preceding line should not start with #, >, -, *, or be blank.
    // Regular expression explanation:
    // ^(?![#>\-*]|\s*$) => line does not start with #, >, -, *, and is not whitespace-only
    // ([^\n]+) => capture the rest of the line
    // \n => newline
    // (A[)\].\s][^\n]*) => option A line starting with A), A., A] or A 
    const regex = /^(?![#>\-*]|\s*$)([^\n]+)\n(A[)\].\s][^\n]*)$/gim;
    
    content = content.replace(regex, '$1\n\n$2');
    
    if (content !== originalContent) {
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`Updated: ${path.basename(filePath)}`);
    } else {
        console.log(`No changes needed for: ${path.basename(filePath)}`);
    }
});
