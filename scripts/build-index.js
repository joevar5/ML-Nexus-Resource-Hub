const fs = require('fs');
const path = require('path');

const IGNORED_DIRS = ['.git', '.github', 'node_modules', 'scripts'];

function extractTitle(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');
        for (let line of lines) {
            line = line.trim();
            if (line.startsWith('# ')) {
                return line.substring(2).trim();
            }
        }
    } catch (err) {
        console.error(`Error reading title from ${filePath}:`, err);
    }
    // Fallback: format filename nicely
    const base = path.basename(filePath, '.md');
    return base
        .split(/[-_]+/)
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function scanDir(dirPath, relativeRoot = '') {
    const items = [];
    let files;
    try {
        files = fs.readdirSync(dirPath);
    } catch (err) {
        console.error(`Error reading directory ${dirPath}:`, err);
        return [];
    }

    for (const file of files) {
        if (IGNORED_DIRS.includes(file)) continue;

        const fullPath = path.join(dirPath, file);
        const relPath = relativeRoot ? path.join(relativeRoot, file) : file;
        
        let stat;
        try {
            stat = fs.statSync(fullPath);
        } catch (err) {
            console.error(`Error statting path ${fullPath}:`, err);
            continue;
        }

        if (stat.isDirectory()) {
            const subItems = scanDir(fullPath, relPath);
            if (subItems.length > 0) {
                items.push({
                    name: file,
                    type: 'directory',
                    children: subItems
                });
            }
        } else if (file.endsWith('.md')) {
            // Skip root level README.md to keep the sidebar clean
            if (!relativeRoot && file.toLowerCase() === 'readme.md') {
                continue;
            }
            const title = extractTitle(fullPath);
            items.push({
                name: title,
                type: 'file',
                path: relPath
            });
        }
    }

    // Sort items: directories first, then files, both alphabetically
    items.sort((a, b) => {
        if (a.type !== b.type) {
            return a.type === 'directory' ? -1 : 1;
        }
        return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
    });

    return items;
}

const rootDir = path.resolve(__dirname, '..');
const structure = scanDir(rootDir);

const outputPath = path.join(rootDir, 'structure.json');
try {
    fs.writeFileSync(outputPath, JSON.stringify(structure, null, 2), 'utf8');
    console.log(`Generated dynamic index in structure.json with ${structure.length} top-level categories.`);
} catch (err) {
    console.error('Error writing structure.json:', err);
    process.exit(1);
}
