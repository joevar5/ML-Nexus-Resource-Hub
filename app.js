let filesTree = [];
let flattenedFiles = [];
let currentFilePath = '';
let activeCategory = null;
let stats = { lessons: '0', designs: '0', projects: '0' };
let completedFiles = new Set(JSON.parse(localStorage.getItem('completed_files') || '[]'));

function saveCompletedFiles() {
    localStorage.setItem('completed_files', JSON.stringify(Array.from(completedFiles)));
}

function toggleFileCompletion(filePath) {
    if (completedFiles.has(filePath)) {
        completedFiles.delete(filePath);
    } else {
        completedFiles.add(filePath);
    }
    saveCompletedFiles();
    updateCategoryProgress();
    updateSidebarCompletionUI();
    updateReaderDoneBtn();
}

function updateCategoryProgress() {
    if (!activeCategory) return;
    const categoryDir = categories.find(c => c.name === activeCategory)?.dir;
    if (!categoryDir) return;

    const totalFiles = flattenedFiles.filter(f => f.path.startsWith(categoryDir) && f.type === 'file');
    const totalCount = totalFiles.length;
    const completedCount = totalFiles.filter(f => completedFiles.has(f.path)).length;

    const progressPercent = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

    document.getElementById('category-progress-text').innerText = `${progressPercent}% (${completedCount}/${totalCount})`;
    document.getElementById('category-progress-bar').style.width = `${progressPercent}%`;
}

function updateSidebarCompletionUI() {
    document.querySelectorAll('.tree-file').forEach(el => {
        const path = el.dataset.filepath;
        if (!path) return;

        const isCompleted = completedFiles.has(path);
        const markDoneBtn = el.querySelector('.mark-done-btn');
        const labelSpan = el.querySelector('.file-label');

        if (markDoneBtn) {
            markDoneBtn.innerText = isCompleted ? 'check_box' : 'check_box_outline_blank';
            if (isCompleted) {
                markDoneBtn.classList.remove('text-on-surface-variant/40');
                markDoneBtn.classList.add('text-primary');
            } else {
                markDoneBtn.classList.add('text-on-surface-variant/40');
                markDoneBtn.classList.remove('text-primary');
            }
        }

        if (labelSpan) {
            if (isCompleted) {
                labelSpan.classList.add('line-through', 'opacity-60');
            } else {
                labelSpan.classList.remove('line-through', 'opacity-60');
            }
        }
    });

    document.querySelectorAll('.tree-folder').forEach(el => {
        const folderpath = el.dataset.folderpath;
        if (!folderpath) return;

        const folderName = folderpath.split('/').pop();
        if (!folderName.startsWith('mod-')) return;

        const descendantFiles = flattenedFiles.filter(f => f.path.startsWith(folderpath + '/'));
        const labelSpan = el.querySelector('.folder-label');
        const bellIcon = el.querySelector('.revise-bell-icon');

        if (descendantFiles.length > 0) {
            const allCompleted = descendantFiles.every(f => completedFiles.has(f.path));

            if (labelSpan) {
                if (allCompleted) {
                    labelSpan.classList.add('line-through', 'opacity-60');
                } else {
                    labelSpan.classList.remove('line-through', 'opacity-60');
                }
            }

            if (bellIcon) {
                if (allCompleted) {
                    bellIcon.classList.remove('hidden');
                } else {
                    bellIcon.classList.add('hidden');
                }
            }
        }
    });
}

function updateReaderDoneBtn() {
    const btn = document.getElementById('mark-done-page-btn');
    const icon = document.getElementById('mark-done-page-icon');
    const text = document.getElementById('mark-done-page-text');
    if (!btn || !icon || !text || !currentFilePath) return;

    const isCompleted = completedFiles.has(currentFilePath);
    if (isCompleted) {
        icon.innerText = 'check_box';
        text.innerText = 'Done';
        btn.className = "flex items-center gap-2 px-6 py-3 bg-primary text-white font-bold rounded-lg transition-all font-label text-sm uppercase tracking-wide border border-primary";
    } else {
        icon.innerText = 'check_box_outline_blank';
        text.innerText = 'Mark as Done';
        btn.className = "flex items-center gap-2 px-6 py-3 bg-surface-container-high hover:bg-primary hover:text-white text-on-surface font-bold rounded-lg transition-all font-label text-sm uppercase tracking-wide border border-outline-variant";
    }
}

const categories = [
    { name: 'Infrastructure', dir: 'AI Infrastructure Engineer', icon: 'memory' },
    { name: 'Projects', dir: 'ML and GenAI Projects', icon: 'terminal' },
    { name: 'System Design', dir: 'ML and GenAI System Design', icon: 'schema' },
    { name: 'Paper Analysis', dir: 'Research Paper Analysis', icon: 'description' }
];

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    setupTheme();
    setupEventListeners();
    loadStructure();
});

// Set up dark / light themes
function setupTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    // Set initial state
    updateThemeToggleButton();

    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.documentElement.classList.contains('dark');
        const newTheme = isDark ? 'light' : 'dark';

        if (newTheme === 'dark') {
            document.documentElement.classList.add('dark');
            document.documentElement.classList.remove('light');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.classList.add('light');
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }
        if (window.mermaid) {
            mermaid.initialize({ theme: newTheme === 'dark' ? 'dark' : 'default' });
        }
        updateThemeToggleButton();
    });
}

function updateThemeToggleButton() {
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const isDark = document.documentElement.classList.contains('dark');
    themeToggleBtn.innerText = isDark ? 'light_mode' : 'dark_mode';
}

// Set up Event Listeners
function setupEventListeners() {
    // Mobile Menu Toggle
    const menuToggle = document.getElementById('menu-toggle-btn');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');

    menuToggle.addEventListener('click', (e) => {
        sidebar.classList.toggle('-translate-x-full');
        overlay.classList.toggle('hidden');
        e.stopPropagation();
    });

    // Close mobile menu on clicking overlay
    overlay.addEventListener('click', () => {
        sidebar.classList.add('-translate-x-full');
        overlay.classList.add('hidden');
    });

    // Handle browser navigation (back/forward)
    window.addEventListener('hashchange', handleRouting);

    // Handle Live Search
    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', (e) => {
        handleSearch(e.target.value);
    });

    // Close mobile menu when links are clicked inside sidebar
    sidebar.addEventListener('click', (e) => {
        if (e.target.closest('.tree-file') || e.target.closest('a') || e.target.closest('button')) {
            sidebar.classList.add('-translate-x-full');
            overlay.classList.add('hidden');
        }
    });

    // Handle scroll progress and scroll-to-top button
    const scrollTopBtn = document.getElementById('scroll-top-btn');
    window.addEventListener('scroll', () => {
        // Progress Bar
        const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
        if (totalHeight > 0) {
            const progress = (window.scrollY / totalHeight) * 100;
            document.getElementById('progress-bar').style.width = `${progress}%`;
        } else {
            document.getElementById('progress-bar').style.width = '0%';
        }

        // Scroll to top visibility
        if (window.scrollY > 300) {
            scrollTopBtn.classList.add('visible');
        } else {
            scrollTopBtn.classList.remove('visible');
        }
    });

    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Intercept relative markdown link clicks to navigate within the SPA
    document.getElementById('markdown-content').addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (!link) return;

        const href = link.getAttribute('href');
        if (!href) return;

        // Handle local page anchor links (prevent changing SPA route hash)
        if (href.startsWith('#')) {
            e.preventDefault();
            const targetId = decodeURIComponent(href.substring(1));
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth' });
            }
            return;
        }

        // Skip external or mailto links
        if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('mailto:')) {
            return;
        }

        // Prevent default normal page navigation
        e.preventDefault();

        // Resolve relative path against current file
        const resolved = resolveRelativePath(currentFilePath, href);
        navigateToResolvedPath(resolved);
    });

    // Selection Handling for ChatGPT Tooltip
    const chatgptTooltip = document.getElementById('chatgpt-tooltip');
    let selectedText = '';

    function handleTextSelection() {
        const selection = window.getSelection();
        const text = selection.toString().trim();

        if (!text) {
            chatgptTooltip.classList.remove('visible');
            selectedText = '';
            return;
        }

        // Only show tooltip in content pages (within markdown-content)
        const markdownContent = document.getElementById('markdown-content');
        if (!markdownContent || !selection.anchorNode || !markdownContent.contains(selection.anchorNode)) {
            chatgptTooltip.classList.remove('visible');
            selectedText = '';
            return;
        }

        selectedText = text;

        // Position the tooltip above the selected text
        try {
            const range = selection.getRangeAt(0);
            const rect = range.getBoundingClientRect();

            // Check if selection bounding box is valid
            if (rect.width > 0 && rect.height > 0) {
                chatgptTooltip.style.left = `${rect.left + rect.width / 2}px`;
                chatgptTooltip.style.top = `${rect.top - 48}px`; // 48px above top of selection
                chatgptTooltip.classList.add('visible');
            } else {
                chatgptTooltip.classList.remove('visible');
            }
        } catch (e) {
            chatgptTooltip.classList.remove('visible');
        }
    }

    document.addEventListener('mouseup', () => {
        // Delay slightly to ensure selection state is fully updated
        setTimeout(handleTextSelection, 10);
    });

    document.addEventListener('keyup', (e) => {
        // Update selection on arrow keys navigation
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Shift'].includes(e.key)) {
            handleTextSelection();
        } else if (e.key === 'Escape') {
            chatgptTooltip.classList.remove('visible');
            window.getSelection().removeAllRanges();
        }
    });

    // Prevent clearing selection immediately when clicking the tooltip itself
    chatgptTooltip.addEventListener('mousedown', (e) => {
        e.preventDefault();
    });

    chatgptTooltip.addEventListener('click', () => {
        if (!selectedText) return;

        const structuredPrompt = `Help me learn the following topic in a structured way:

${selectedText}

Please:
1. Explain it in simple English.
2. Break it down step by step.
3. Explain how it works internally.
4. Clear common doubts and misconceptions.
5. Provide practical examples.
6. Highlight important concepts and key takeaways.
7. Ask me a few questions to test my understanding.
8. Suggest what I should learn next.

Assume I am learning this for practical use and interview preparation.`;

        const chatgptUrl = `https://chatgpt.com/?q=${encodeURIComponent(structuredPrompt)}`;
        window.open(chatgptUrl, '_blank');

        // Clear selection and hide tooltip
        window.getSelection().removeAllRanges();
        chatgptTooltip.classList.remove('visible');
    });

    // Handle page completion toggle button
    const donePageBtn = document.getElementById('mark-done-page-btn');
    if (donePageBtn) {
        donePageBtn.addEventListener('click', () => {
            if (currentFilePath) {
                toggleFileCompletion(currentFilePath);
            }
        });
    }
}

// Select learning path category in sidebar
function selectCategory(categoryName) {
    const cat = categories.find(c => c.name === categoryName);
    if (!cat) return;

    activeCategory = categoryName;

    // Reset About the Author link styles
    document.getElementById('about-author-link').className = 'flex items-center gap-md p-3 text-on-surface-variant hover:bg-surface-container-high rounded-lg transition-colors';

    // Highlight active category tab in sidebar
    const catBtns = document.querySelectorAll('.cat-btn');
    catBtns.forEach(btn => {
        if (btn.dataset.category === categoryName) {
            btn.className = "cat-btn w-full flex items-center gap-md p-3 rounded-lg bg-primary-container text-on-primary-container font-medium transition-all text-left";
        } else {
            btn.className = "cat-btn w-full flex items-center gap-md p-3 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors text-left";
        }
    });

    // Filter filesTree and render sidebar files navigation
    const targetNode = filesTree.find(node => node.name === cat.dir);
    if (targetNode && targetNode.children) {
        renderSidebar(targetNode.children, document.getElementById('sidebar-menu'), cat.dir);
    } else {
        document.getElementById('sidebar-menu').innerHTML = `
                    <div class="p-2 text-on-surface-variant/60 text-xs">
                        No resources found in this path.
                    </div>
                `;
    }
    updateCategoryProgress();
}

// Navigate dynamically to the first file of a category via URL hash
function navigateToCategory(categoryName) {
    const cat = categories.find(c => c.name === categoryName);
    if (!cat) return;

    const targetNode = filesTree.find(node => node.name === cat.dir);
    if (targetNode && targetNode.children) {
        const firstFile = findFirstFile(targetNode.children);
        if (firstFile) {
            window.location.hash = encodeURIComponent(firstFile.path);
        }
    }
}

// Recursive search for the first readable markdown file in category children
function findFirstFile(nodes) {
    // 1. Search for a readme file recursively
    const readme = findReadmeRecursively(nodes);
    if (readme) return readme;

    // 2. If no readme, return the first file we encounter
    return findFirstFileRecursively(nodes);
}

function findReadmeRecursively(nodes) {
    // First check current level
    const readme = nodes.find(n => n.type === 'file' && n.path.toLowerCase().endsWith('readme.md'));
    if (readme) return readme;

    // Then check subdirectories
    for (const n of nodes) {
        if (n.type === 'directory' && n.children) {
            const f = findReadmeRecursively(n.children);
            if (f) return f;
        }
    }
    return null;
}

function findFirstFileRecursively(nodes) {
    for (const n of nodes) {
        if (n.type === 'file') return n;
        if (n.type === 'directory' && n.children) {
            const f = findFirstFileRecursively(n.children);
            if (f) return f;
        }
    }
    return null;
}

// Resolve relative paths in links/images
function resolveRelativePath(basePath, relativePath) {
    relativePath = decodeURIComponent(relativePath);

    const parts = basePath.split('/');
    parts.pop(); // Remove filename to get parent dir

    const relParts = relativePath.split('/');
    for (const part of relParts) {
        if (part === '.' || part === '') {
            continue;
        } else if (part === '..') {
            parts.pop();
        } else {
            parts.push(part);
        }
    }
    return parts.join('/');
}

// Route internal link click to dynamic routing hash
function navigateToResolvedPath(resolvedPath) {
    const exactFile = flattenedFiles.find(f => f.path === resolvedPath);
    if (exactFile) {
        window.location.hash = encodeURIComponent(resolvedPath);
        return;
    }

    // Fallbacks for folders
    const readmeAlternatives = [
        resolvedPath + '/readme.md',
        resolvedPath + '/README.md',
        resolvedPath + '/Readme.md',
        resolvedPath + '/index.html'
    ];

    for (const rPath of readmeAlternatives) {
        const readmeFile = flattenedFiles.find(f => f.path === rPath);
        if (readmeFile) {
            window.location.hash = encodeURIComponent(rPath);
            return;
        }
    }

    const dirPrefix = resolvedPath.endsWith('/') ? resolvedPath : resolvedPath + '/';
    const childFile = flattenedFiles.find(f => f.path.startsWith(dirPrefix));
    if (childFile) {
        window.location.hash = encodeURIComponent(childFile.path);
        return;
    }

    console.warn(`Could not resolve route for: ${resolvedPath}`);
    window.location.hash = encodeURIComponent(resolvedPath);
}

// Fetch structure.json and load app index
async function loadStructure() {
    try {
        const response = await fetch('structure.json');
        if (!response.ok) throw new Error('Failed to fetch structure index');
        filesTree = await response.json();

        flattenTree(filesTree);
        calculateStats();

        // Select first category by default
        selectCategory('Infrastructure');

        handleRouting();
    } catch (err) {
        console.error('Error initializing structure:', err);
        document.getElementById('sidebar-menu').innerHTML = `
                    <div class="p-2 text-error text-xs font-semibold">
                        Failed to build resource navigation. Run the indexing script.
                    </div>
                `;
    }
}

// Flatten file nodes
function flattenTree(nodes) {
    for (const node of nodes) {
        if (node.type === 'file') {
            flattenedFiles.push(node);
        } else if (node.type === 'directory' && node.children) {
            flattenTree(node.children);
        }
    }
}

// Calculate dynamic repository stats
function calculateStats() {
    let lessonsCount = 0;
    let designsCount = 0;
    let projectsCount = 0;

    flattenedFiles.forEach(f => {
        const lowerPath = f.path.toLowerCase();
        if (lowerPath.includes('lesson') || lowerPath.includes('lecture')) lessonsCount++;
        else if (lowerPath.includes('system design') || lowerPath.includes('mlsd')) designsCount++;
        else if (lowerPath.includes('project')) projectsCount++;
    });

    stats.lessons = lessonsCount || '15+';
    stats.designs = designsCount || '50+';
    stats.projects = projectsCount || '5+';

    // Inject count values into dashboard cards
    const projectsDesc = document.getElementById('projects-desc');
    if (projectsDesc) {
        projectsDesc.innerText = `From RAG pipelines to fine-tuning Llama-3, build end-to-end production examples. (${stats.projects} ready projects)`;
    }
    const designsDesc = document.getElementById('designs-desc');
    if (designsDesc) {
        designsDesc.innerText = `Design scalable architectures for recommendation engines and large-scale vector search. (${stats.designs} modules)`;
    }
}

// Render recursive tree sidebar navigation
function renderSidebar(nodes, container, parentPath = '') {
    container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.className = "space-y-1";

    nodes.forEach(node => {
        const item = document.createElement('div');
        item.className = 'tree-node';

        const currentPath = parentPath ? `${parentPath}/${node.name}` : node.name;

        if (node.type === 'directory') {
            if (node.name.startsWith('Phase ') || node.name === 'Introduction') {
                const isThisPhaseActive = currentFilePath && currentFilePath.startsWith(currentPath);
                const isDefaultExpanded = (node.name === 'Introduction' || (node.name.includes('Phase 1') && (!currentFilePath || !currentFilePath.startsWith('AI Infrastructure Engineer'))));
                const isExpanded = isThisPhaseActive || isDefaultExpanded;

                const headerDiv = document.createElement('div');
                headerDiv.className = 'sidebar-section-header px-2 py-3 mt-4 text-xs font-bold uppercase tracking-wider text-primary border-b border-outline-variant/30 mb-2 first:mt-0 select-none cursor-pointer flex items-center justify-between hover:text-primary transition-all';
                headerDiv.innerHTML = `
                    <span>${node.name}</span>
                    <span class="material-symbols-outlined text-xs phase-chevron transition-transform">${isExpanded ? 'expand_more' : 'chevron_right'}</span>
                `;

                const childrenDiv = document.createElement('div');
                childrenDiv.className = `phase-container space-y-1 mt-1 ${isExpanded ? '' : 'hidden'}`;
                childrenDiv.dataset.phasepath = currentPath;

                headerDiv.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    const wasHidden = childrenDiv.classList.contains('hidden');

                    document.querySelectorAll('.phase-container').forEach(container => {
                        container.classList.add('hidden');
                        const header = container.previousElementSibling;
                        if (header) {
                            const chevron = header.querySelector('.phase-chevron');
                            if (chevron) chevron.innerText = 'chevron_right';
                        }
                    });

                    if (wasHidden) {
                        childrenDiv.classList.remove('hidden');
                        const chevron = headerDiv.querySelector('.phase-chevron');
                        if (chevron) chevron.innerText = 'expand_more';
                    }
                });

                item.appendChild(headerDiv);
                item.appendChild(childrenDiv);

                renderSidebar(node.children, childrenDiv, currentPath);
            } else {
                const folderDiv = document.createElement('div');
                folderDiv.className = 'tree-folder flex items-center justify-between gap-md p-2 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer text-sm font-label';
                folderDiv.dataset.folderpath = currentPath;

                const isModFolder = node.name.startsWith('mod-');
                let isAllCompleted = false;
                let labelStyle = '';
                let bellStyle = 'hidden';

                if (isModFolder) {
                    const descendantFiles = flattenedFiles.filter(f => f.path.startsWith(currentPath + '/'));
                    isAllCompleted = descendantFiles.length > 0 && descendantFiles.every(f => completedFiles.has(f.path));
                    labelStyle = isAllCompleted ? 'line-through opacity-60' : '';
                    bellStyle = isAllCompleted ? '' : 'hidden';
                }

                const bellHTML = isModFolder ? `
                            <span class="revise-bell-icon material-symbols-outlined text-md text-amber-500 dark:text-amber-400 hover:scale-110 transition-transform cursor-pointer flex-shrink-0 ${bellStyle} ml-1" title="Revise Now">notifications</span>
                        ` : '';

                folderDiv.innerHTML = `
                            <div class="flex items-center gap-2 overflow-hidden flex-1">
                                <span class="material-symbols-outlined text-md text-on-surface-variant/80 flex-shrink-0">folder</span>
                                <span class="folder-label whitespace-normal break-words text-left leading-tight ${labelStyle}">${node.name}</span>
                            </div>
                            <div class="flex items-center gap-2 flex-shrink-0">
                                ${bellHTML}
                                <span class="material-symbols-outlined text-xs tree-folder-chevron transition-transform flex-shrink-0">chevron_right</span>
                            </div>
                        `;

                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children hidden pl-3 border-l border-outline-variant/40 ml-4 space-y-1 mt-1';

                folderDiv.addEventListener('click', (e) => {
                    childrenDiv.classList.toggle('hidden');
                    const chevron = folderDiv.querySelector('.tree-folder-chevron');
                    chevron.classList.toggle('rotate-90');
                    e.stopPropagation();
                });

                const bellIcon = folderDiv.querySelector('.revise-bell-icon');
                if (bellIcon) {
                    bellIcon.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();

                        // Find README file under this folder
                        let readmeFile = node.children.find(child => child.type === 'file' && child.path.toLowerCase().endsWith('readme.md'));
                        if (!readmeFile) {
                            const descendantFiles = flattenedFiles.filter(f => f.path.startsWith(currentPath + '/'));
                            readmeFile = descendantFiles.find(f => f.path.toLowerCase().endsWith('readme.md'));
                        }

                        if (readmeFile) {
                            window.location.hash = encodeURIComponent(readmeFile.path);
                        }
                    });
                }

                item.appendChild(folderDiv);
                item.appendChild(childrenDiv);

                renderSidebar(node.children, childrenDiv, currentPath);
            }
        } else {
            const fileLink = document.createElement('a');
            const isCompleted = completedFiles.has(node.path);

            fileLink.className = 'tree-file flex items-start gap-2 p-2 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer text-sm font-label';
            fileLink.href = `#${encodeURIComponent(node.path)}`;
            fileLink.dataset.filepath = node.path;

            const checkColor = isCompleted ? 'text-primary' : 'text-on-surface-variant/40';
            const labelStyle = isCompleted ? 'line-through opacity-60' : '';

            fileLink.innerHTML = `
                        <span class="mark-done-btn material-symbols-outlined text-md flex-shrink-0 mt-0.5 hover:text-primary transition-colors ${checkColor}">${isCompleted ? 'check_box' : 'check_box_outline_blank'}</span>
                        <span class="file-label whitespace-normal break-words text-left leading-tight ${labelStyle}">${node.name}</span>
                    `;

            const checkBtn = fileLink.querySelector('.mark-done-btn');
            checkBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                toggleFileCompletion(node.path);
            });

            fileLink.addEventListener('click', () => {
                document.querySelectorAll('.tree-file').forEach(el => {
                    el.className = 'tree-file flex items-start gap-2 p-2 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer text-sm font-label';
                });
                fileLink.className = 'tree-file active flex items-start gap-2 p-2 rounded-lg bg-primary-container text-on-primary-container font-medium transition-all cursor-pointer text-sm font-label';
            });

            item.appendChild(fileLink);
        }
        wrapper.appendChild(item);
    });
    container.appendChild(wrapper);
}

// Routing Handler
function handleRouting() {
    const hash = window.location.hash.substring(1);
    const searchInput = document.getElementById('search-input');

    if (hash && searchInput.value) {
        searchInput.value = '';
    }

    if (!hash || decodeURIComponent(hash) === 'README.md') {
        goHome();
        return;
    }

    const decodedPath = decodeURIComponent(hash);

    if (decodedPath === 'about') {
        showAuthorPage();
        return;
    }

    loadFile(decodedPath);
}

// Go home / Dashboard view
function goHome() {
    window.location.hash = '';
    currentFilePath = 'README.md';

    document.getElementById('home-view').style.display = 'block';
    document.getElementById('reader-view').style.display = 'none';
    document.getElementById('search-view').style.display = 'none';
    document.getElementById('error-view').style.display = 'none';
    document.getElementById('author-view').style.display = 'none';
    document.getElementById('sidebar-tree-container').style.display = 'none';

    document.querySelectorAll('.tree-file').forEach(el => {
        el.className = 'tree-file flex items-start gap-2 p-2 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer text-sm font-label';
    });

    // Reset About the Author link styles
    document.getElementById('about-author-link').className = 'flex items-center gap-md p-3 text-on-surface-variant hover:bg-surface-container-high rounded-lg transition-colors';

    document.getElementById('progress-bar').style.width = '0%';
}

// Display Author Page
function showAuthorPage() {
    currentFilePath = '';

    document.getElementById('home-view').style.display = 'none';
    document.getElementById('reader-view').style.display = 'none';
    document.getElementById('search-view').style.display = 'none';
    document.getElementById('error-view').style.display = 'none';
    document.getElementById('author-view').style.display = 'block';
    document.getElementById('sidebar-tree-container').style.display = 'none';

    // Reset files highlights
    document.querySelectorAll('.tree-file').forEach(el => {
        el.className = 'tree-file flex items-start gap-2 p-2 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer text-sm font-label';
    });

    // Highlight active category tab in sidebar
    const catBtns = document.querySelectorAll('.cat-btn');
    catBtns.forEach(btn => {
        btn.className = "cat-btn w-full flex items-center gap-md p-3 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors text-left";
    });

    // Highlight About the Author button in sidebar footer
    document.getElementById('about-author-link').className = 'flex items-center gap-md p-3 bg-primary-container text-on-primary-container rounded-lg transition-colors';

    document.getElementById('progress-bar').style.width = '0%';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Parsing custom alert quote blocks
function parseAlertBlocks(markdown) {
    const alertRegex = /^>\s*\[\!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:^>.*\n?)*)/gim;
    return markdown.replace(alertRegex, (match, type, content) => {
        const cleanContent = content.replace(/^>\s?/gm, '').trim();
        const cleanType = type.toLowerCase();

        let iconName = 'info';
        if (cleanType === 'note') iconName = 'info';
        else if (cleanType === 'tip') iconName = 'lightbulb';
        else if (cleanType === 'important') iconName = 'priority_high';
        else if (cleanType === 'warning') iconName = 'warning';
        else if (cleanType === 'caution') iconName = 'report';

        return `<div class="alert-box ${cleanType}">
<div class="alert-title">
<span class="material-symbols-outlined text-sm">${iconName}</span>
<span>${type}</span>
</div>
<div class="alert-body">${marked.parse(cleanContent)}</div>
</div>\n\n`;
    });
}

// Copy buttons code blocks
function addCopyButtons() {
    const preBlocks = document.querySelectorAll('.markdown-body pre');
    preBlocks.forEach(pre => {
        if (pre.parentNode.classList.contains('code-wrapper')) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'code-wrapper';

        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        const button = document.createElement('button');
        button.className = 'copy-code-btn';
        button.innerText = 'Copy';
        button.addEventListener('click', () => {
            const code = pre.querySelector('code');
            navigator.clipboard.writeText(code.innerText).then(() => {
                button.innerText = 'Copied!';
                setTimeout(() => { button.innerText = 'Copy'; }, 2000);
            });
        });
        wrapper.appendChild(button);
    });
}

// ── Zoom Lightbox for Images & Mermaid Diagrams ──
function initZoomLightbox() {
    const lightbox = document.getElementById('zoom-lightbox');
    const contentWrap = document.getElementById('zoom-lightbox-content');
    const zoomLevel = document.getElementById('zoom-level');
    const magLens = document.getElementById('mag-lens');
    const MAG_SCALE = 2.5;
    const LENS_SIZE = 180;
    let scale = 1, translateX = 0, translateY = 0;
    let isDragging = false, startX = 0, startY = 0;

    function applyTransform() {
        contentWrap.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
        zoomLevel.textContent = Math.round(scale * 100) + '%';
    }
    function openLightbox(el) {
        contentWrap.innerHTML = '';
        let clone;
        if (el.tagName === 'IMG') {
            clone = document.createElement('img');
            clone.src = el.src;
            clone.alt = el.alt || '';
        } else {
            clone = el.cloneNode(true);
        }
        contentWrap.appendChild(clone);
        scale = 1; translateX = 0; translateY = 0;
        applyTransform();
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }

    // ── Magnifier: convert element to image URL for background ──
    function getImageURL(el) {
        if (el.tagName === 'IMG') return el.src;
        // For SVG (mermaid), serialize to data URL
        const svgEl = el.tagName === 'svg' ? el : el.querySelector('svg');
        if (!svgEl) return null;
        const clone = svgEl.cloneNode(true);
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        
        // Inject current theme's text color into the SVG so it is preserved in the data URL image
        const computedColor = getComputedStyle(document.documentElement).getPropertyValue('--color-on-background').trim();
        if (computedColor) {
            const svgStyle = document.createElementNS('http://www.w3.org/2000/svg', 'style');
            svgStyle.textContent = `.messageText, .labelText, .loopText { fill: ${computedColor} !important; }`;
            clone.appendChild(svgStyle);
        }

        const svgStr = new XMLSerializer().serializeToString(clone);
        return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgStr);
    }

    // ── Attach magnifier to each zoomable element ──
    function attachMagnifier(el) {
        let imgURL = null;
        let naturalW = 0, naturalH = 0;

        el.addEventListener('mouseenter', () => {
            imgURL = getImageURL(el);
            if (!imgURL) return;
            // Get rendered size
            const rect = el.getBoundingClientRect();
            naturalW = rect.width;
            naturalH = rect.height;
            magLens.style.backgroundImage = `url("${imgURL}")`;
            magLens.style.backgroundSize = `${naturalW * MAG_SCALE}px ${naturalH * MAG_SCALE}px`;
            magLens.classList.add('active');
        });

        el.addEventListener('mousemove', (e) => {
            if (!imgURL) return;
            const rect = el.getBoundingClientRect();
            // Cursor position relative to the element (0-1)
            const ratioX = (e.clientX - rect.left) / rect.width;
            const ratioY = (e.clientY - rect.top) / rect.height;
            // Background position so the cursor area is centered in the lens
            const bgX = ratioX * naturalW * MAG_SCALE - LENS_SIZE / 2;
            const bgY = ratioY * naturalH * MAG_SCALE - LENS_SIZE / 2;
            magLens.style.backgroundPosition = `-${bgX}px -${bgY}px`;
            // Position the lens near the cursor
            magLens.style.left = `${e.clientX - LENS_SIZE / 2}px`;
            magLens.style.top = `${e.clientY - LENS_SIZE / 2}px`;
        });

        el.addEventListener('mouseleave', () => {
            magLens.classList.remove('active');
            imgURL = null;
        });

        // Click opens the full lightbox
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            magLens.classList.remove('active');
            openLightbox(el);
        });
    }

    // Attach to all images
    document.querySelectorAll('#markdown-content img').forEach(img => attachMagnifier(img));
    // Attach to all mermaid SVGs
    document.querySelectorAll('#markdown-content .mermaid svg').forEach(svg => {
        svg.style.cursor = 'none';
        attachMagnifier(svg);
    });

    // Close lightbox
    document.getElementById('zoom-close').addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

    // Zoom buttons
    document.getElementById('zoom-in-btn').addEventListener('click', (e) => {
        e.stopPropagation(); scale = Math.min(scale * 1.3, 8); applyTransform();
    });
    document.getElementById('zoom-out-btn').addEventListener('click', (e) => {
        e.stopPropagation(); scale = Math.max(scale / 1.3, 0.2); applyTransform();
    });
    document.getElementById('zoom-reset-btn').addEventListener('click', (e) => {
        e.stopPropagation(); scale = 1; translateX = 0; translateY = 0; applyTransform();
    });

    // Mouse wheel zoom
    lightbox.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        scale = Math.min(Math.max(scale * delta, 0.2), 8);
        applyTransform();
    }, { passive: false });

    // Drag to pan
    lightbox.addEventListener('mousedown', (e) => {
        if (e.target.closest('.zoom-toolbar') || e.target.closest('.zoom-close-btn')) return;
        isDragging = true; startX = e.clientX - translateX; startY = e.clientY - translateY;
        lightbox.classList.add('dragging');
    });
    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        translateX = e.clientX - startX; translateY = e.clientY - startY;
        applyTransform();
    });
    window.addEventListener('mouseup', () => {
        isDragging = false; lightbox.classList.remove('dragging');
    });
}

// Local image path resolution
function fixRelativeImages() {
    const images = document.querySelectorAll('#markdown-content img');
    images.forEach(img => {
        const src = img.getAttribute('src');
        if (!src) return;

        if (src.startsWith('http://') || src.startsWith('https://') || src.startsWith('data:')) {
            return;
        }

        const resolvedSrc = resolveRelativePath(currentFilePath, src);
        img.setAttribute('src', resolvedSrc);
    });
}

// Highlight active sidebar file path & expand its folders recursively
function highlightActiveFileInSidebar(filePath) {
    document.querySelectorAll('.tree-file').forEach(el => {
        if (el.dataset.filepath === filePath) {
            el.className = "tree-file active flex items-start gap-2 p-2 rounded-lg bg-primary-container text-on-primary-container font-medium transition-all cursor-pointer text-sm font-label";

            let parent = el.closest('.tree-children');
            while (parent) {
                parent.classList.remove('hidden');
                const parentNode = parent.closest('.tree-node');
                if (parentNode) {
                    const chevron = parentNode.querySelector('.tree-folder-chevron');
                    if (chevron) chevron.classList.add('rotate-90');
                }
                parent = parentNode ? parentNode.parentElement.closest('.tree-children') : null;
            }

            const phaseContainer = el.closest('.phase-container');
            if (phaseContainer && phaseContainer.classList.contains('hidden')) {
                document.querySelectorAll('.phase-container').forEach(pc => {
                    pc.classList.add('hidden');
                    const header = pc.previousElementSibling;
                    if (header) {
                        const chevron = header.querySelector('.phase-chevron');
                        if (chevron) chevron.innerText = 'chevron_right';
                    }
                });

                phaseContainer.classList.remove('hidden');
                const header = phaseContainer.previousElementSibling;
                if (header) {
                    const chevron = header.querySelector('.phase-chevron');
                    if (chevron) chevron.innerText = 'expand_more';
                }
            }
        } else {
            el.className = "tree-file flex items-start gap-2 p-2 rounded-lg text-on-surface-variant hover:bg-surface-container-high transition-colors cursor-pointer text-sm font-label";
        }
    });
}

// Load Markdown file and update UI view panels
// ── Interactive Quiz Engine ──
// Detects quiz patterns in rendered markdown and makes them clickable with radio buttons
function makeQuizInteractive() {
    const container = document.getElementById('markdown-content');
    if (!container) return;

    // Find all h3 headings that look like quiz questions
    const questionHeadings = Array.from(container.querySelectorAll('h3')).filter(h3 => {
        return /question\s+\d+/i.test(h3.textContent);
    });

    if (questionHeadings.length === 0) return;

    // ── Parse quiz metadata from the page content ──
    const fullText = container.textContent;
    const totalQuestions = questionHeadings.length;

    // Try to parse passing score: "Passing Score: 70% (20/28 correct)" or "80% (40/50 correct)"
    let passingPercent = 70; // default
    const passingMatch = fullText.match(/Passing\s+Score:?\s*(\d+)%/i);
    if (passingMatch) passingPercent = parseInt(passingMatch[1]);

    const passingCount = Math.ceil((passingPercent / 100) * totalQuestions);

    // ── Quiz state ──
    let quizState = {
        correct: 0,
        wrong: 0,
        total: totalQuestions,
        answered: 0,
        questionStates: {} // track per-question state for retry
    };

    // ── Load saved quiz state if any ──
    const storageKey = `quiz_answers_${currentFilePath}`;
    let savedAnswers = {};
    try {
        const saved = localStorage.getItem(storageKey);
        if (saved) {
            savedAnswers = JSON.parse(saved);
        }
    } catch (e) {
        console.error('Failed to load quiz progress', e);
    }

    // ── Build scoreboard ──
    const scoreboard = document.createElement('div');
    scoreboard.className = 'quiz-scoreboard';
    scoreboard.id = 'quiz-scoreboard';
    scoreboard.innerHTML = `
                <div class="quiz-scoreboard-left">
                    <div class="quiz-score-pct" id="quiz-score-pct">0%</div>
                    <div class="quiz-score-info">
                        <div class="quiz-score-title" id="quiz-score-title">Quiz in Progress</div>
                        <div class="quiz-score-subtitle">Pass: ${passingPercent}% (${passingCount}/${totalQuestions})</div>
                    </div>
                </div>
                <div class="quiz-scoreboard-right">
                    <div class="quiz-score-pills">
                        <span class="quiz-pill quiz-pill-correct">
                            <span class="material-symbols-outlined" style="font-size:16px">check_circle</span>
                            <span id="quiz-sb-correct">0</span>
                        </span>
                        <span class="quiz-pill quiz-pill-wrong">
                            <span class="material-symbols-outlined" style="font-size:16px">cancel</span>
                            <span id="quiz-sb-wrong">0</span>
                        </span>
                        <span class="quiz-pill quiz-pill-remaining">
                            <span class="material-symbols-outlined" style="font-size:16px">radio_button_unchecked</span>
                            <span id="quiz-sb-remaining">${totalQuestions}</span>
                        </span>
                    </div>
                    <button class="quiz-reset-all-btn" id="quiz-reset-all-btn">
                        <span class="material-symbols-outlined" style="font-size:16px">restart_alt</span>
                        Reset All
                    </button>
                </div>
            `;

    // Append scoreboard to the end of the container to enable sticky bottom behavior
    container.appendChild(scoreboard);

    // ── Scoreboard update function ──
    function updateScoreboard() {
        const pct = quizState.total > 0 ? Math.round((quizState.correct / quizState.total) * 100) : 0;

        // Update percentage display
        const pctEl = document.getElementById('quiz-score-pct');
        pctEl.textContent = `${pct}%`;

        // Color the percentage based on status
        if (quizState.answered === 0) {
            pctEl.className = 'quiz-score-pct';
        } else if (quizState.correct >= passingCount) {
            pctEl.className = 'quiz-score-pct quiz-score-pct-passed';
        } else if (quizState.answered === quizState.total && quizState.correct < passingCount) {
            pctEl.className = 'quiz-score-pct quiz-score-pct-failed';
        } else {
            pctEl.className = 'quiz-score-pct';
        }

        // Update pills
        document.getElementById('quiz-sb-correct').textContent = quizState.correct;
        document.getElementById('quiz-sb-wrong').textContent = quizState.wrong;
        document.getElementById('quiz-sb-remaining').textContent = quizState.total - quizState.answered;

        // Update title and pass/fail status
        const titleEl = document.getElementById('quiz-score-title');
        if (quizState.answered === quizState.total) {
            if (quizState.correct >= passingCount) {
                titleEl.innerHTML = `<span class="quiz-status-badge quiz-status-passed">
                            <span class="material-symbols-outlined" style="font-size:16px">emoji_events</span> Passed!
                        </span> ${quizState.correct}/${quizState.total} correct`;
            } else {
                titleEl.innerHTML = `<span class="quiz-status-badge quiz-status-failed">
                            <span class="material-symbols-outlined" style="font-size:16px">close</span> Not Passed
                        </span> ${quizState.correct}/${quizState.total} correct`;
            }
        } else if (quizState.answered > 0) {
            titleEl.textContent = `${quizState.correct}/${quizState.answered} correct so far`;
        } else {
            titleEl.textContent = 'Quiz in Progress';
        }
    }

    // ── Process each question ──
    const allRetryBtns = [];
    const allQuestionOptionSets = [];

    questionHeadings.forEach((h3, qIndex) => {
        const questionElements = [];
        let sibling = h3.nextElementSibling;
        while (sibling) {
            if (sibling.tagName === 'H3' || sibling.tagName === 'H2' || sibling.tagName === 'HR') break;
            questionElements.push(sibling);
            sibling = sibling.nextElementSibling;
        }

        if (questionElements.length === 0) return;

        let correctAnswer = '';
        let answerEl = null;
        let explanationEls = [];
        let questionTextEls = [];
        let rawOptionLines = [];
        let optionSourceEls = [];
        let foundAnswer = false;

        for (const elem of questionElements) {
            const text = elem.textContent.trim();

            if (/^(Correct\s+)?Answer:?\s*[A-D]/i.test(text)) {
                answerEl = elem;
                const match = text.match(/(?:Correct\s+)?Answer:?\s*([A-D])/i);
                if (match) correctAnswer = match[1].toUpperCase();
                foundAnswer = true;
                continue;
            }

            if (foundAnswer) {
                explanationEls.push(elem);
                continue;
            }

            const innerHtml = elem.innerHTML || '';
            const htmlLines = innerHtml.split(/<br\s*\/?>/i).map(l => l.trim()).filter(l => l);
            let hasOptions = false;

            for (const line of htmlLines) {
                const temp = document.createElement('span');
                temp.innerHTML = line;
                const lineText = temp.textContent.trim();

                if (/^[A-D][)\].]/i.test(lineText)) {
                    hasOptions = true;
                    rawOptionLines.push({ text: lineText, html: line });
                }
            }

            if (hasOptions) {
                optionSourceEls.push(elem);
            } else if (rawOptionLines.length === 0) {
                questionTextEls.push(elem);
            }
        }

        if (rawOptionLines.length === 0 || !correctAnswer) return;

        // Build interactive options
        const optionsContainer = document.createElement('div');
        optionsContainer.className = 'quiz-options-container';

        rawOptionLines.forEach(({ text, html }) => {
            const letterMatch = text.match(/^([A-D])[)\].]\s*(.*)/is);
            if (!letterMatch) return;

            const letter = letterMatch[1].toUpperCase();
            const htmlMatch = html.match(/^[A-D][)\].]\s*(.*)/is);
            const optionContent = htmlMatch ? htmlMatch[1].trim() : letterMatch[2].trim();
            const isCorrect = letter === correctAnswer;

            const optionDiv = document.createElement('div');
            optionDiv.className = 'quiz-option';
            optionDiv.dataset.letter = letter;
            optionDiv.dataset.correct = isCorrect.toString();

            optionDiv.innerHTML = `
                        <span class="quiz-option-radio"></span>
                        <span class="quiz-option-letter">${letter}</span>
                        <span class="quiz-option-text">${optionContent}</span>
                        <span class="quiz-option-result-icon">
                            <span class="material-symbols-outlined" style="font-size:18px">${isCorrect ? 'check' : 'close'}</span>
                        </span>
                    `;

            optionsContainer.appendChild(optionDiv);
        });

        optionSourceEls.forEach(el => { el.style.display = 'none'; });

        // Explanation wrapper
        const explanationWrapper = document.createElement('div');
        explanationWrapper.className = 'quiz-explanation';
        const explanationInner = document.createElement('div');
        explanationInner.className = 'quiz-explanation-inner';

        let explanationHTML = '';
        if (answerEl) {
            explanationHTML += `<strong>✓ Correct Answer: ${correctAnswer}</strong><br>`;
        }
        explanationEls.forEach(el => {
            explanationHTML += el.innerHTML;
            el.style.display = 'none';
        });
        explanationInner.innerHTML = explanationHTML;
        explanationWrapper.appendChild(explanationInner);

        if (answerEl) answerEl.style.display = 'none';

        // Retry button
        const retryBtn = document.createElement('button');
        retryBtn.className = 'quiz-retry-btn';
        retryBtn.innerHTML = `<span class="material-symbols-outlined" style="font-size:15px">refresh</span> Try Again`;
        retryBtn.style.display = 'none';
        allRetryBtns.push(retryBtn);

        // Insert elements
        const lastQuestionTextEl = questionTextEls.length > 0 ? questionTextEls[questionTextEls.length - 1] : h3;
        lastQuestionTextEl.after(optionsContainer);
        optionsContainer.after(explanationWrapper);
        explanationWrapper.after(retryBtn);

        // Click handler
        const allOptions = optionsContainer.querySelectorAll('.quiz-option');
        allQuestionOptionSets.push({ allOptions, explanationWrapper, retryBtn, qIndex });

        function selectOption(opt, save = true) {
            const isCorrect = opt.dataset.correct === 'true';

            // If this question was previously answered (via retry), undo the previous answer
            if (quizState.questionStates[qIndex]) {
                const prev = quizState.questionStates[qIndex];
                if (prev === 'correct') quizState.correct--;
                else quizState.wrong--;
                quizState.answered--;
            }

            opt.classList.add('quiz-option-selected');
            allOptions.forEach(o => o.classList.add('quiz-option-disabled'));

            if (isCorrect) {
                opt.classList.add('quiz-option-correct');
                quizState.correct++;
                quizState.questionStates[qIndex] = 'correct';
            } else {
                opt.classList.add('quiz-option-wrong');
                quizState.wrong++;
                quizState.questionStates[qIndex] = 'wrong';
                allOptions.forEach(o => {
                    if (o.dataset.correct === 'true') {
                        o.classList.add('quiz-option-correct');
                    }
                });
            }

            quizState.answered++;
            updateScoreboard();

            explanationWrapper.classList.add('quiz-explanation-visible');
            retryBtn.style.display = 'inline-flex';

            if (save) {
                savedAnswers[qIndex] = opt.dataset.letter;
                try {
                    localStorage.setItem(storageKey, JSON.stringify(savedAnswers));
                } catch (e) {
                    console.error('Failed to save quiz progress', e);
                }
            }
        }

        allOptions.forEach(opt => {
            opt.addEventListener('click', () => {
                if (opt.classList.contains('quiz-option-disabled')) return;
                selectOption(opt, true);
            });
        });

        // Retry handler
        retryBtn.addEventListener('click', () => {
            // Undo this question's score
            if (quizState.questionStates[qIndex]) {
                const prev = quizState.questionStates[qIndex];
                if (prev === 'correct') quizState.correct--;
                else quizState.wrong--;
                quizState.answered--;
                delete quizState.questionStates[qIndex];
                updateScoreboard();
            }

            allOptions.forEach(o => {
                o.classList.remove('quiz-option-disabled', 'quiz-option-correct', 'quiz-option-wrong', 'quiz-option-selected');
            });
            explanationWrapper.classList.remove('quiz-explanation-visible');
            retryBtn.style.display = 'none';

            // Remove from saved answers and save to localStorage
            delete savedAnswers[qIndex];
            try {
                localStorage.setItem(storageKey, JSON.stringify(savedAnswers));
            } catch (e) {
                console.error('Failed to save quiz progress after retry', e);
            }
        });

        // Restore saved option if any
        if (savedAnswers[qIndex] !== undefined) {
            const savedLetter = savedAnswers[qIndex];
            const savedOpt = Array.from(allOptions).find(o => o.dataset.letter === savedLetter);
            if (savedOpt) {
                selectOption(savedOpt, false);
            }
        }
    });

    // ── Reset All button handler ──
    document.getElementById('quiz-reset-all-btn').addEventListener('click', () => {
        quizState = {
            correct: 0,
            wrong: 0,
            total: totalQuestions,
            answered: 0,
            questionStates: {}
        };

        allQuestionOptionSets.forEach(({ allOptions, explanationWrapper, retryBtn }) => {
            allOptions.forEach(o => {
                o.classList.remove('quiz-option-disabled', 'quiz-option-correct', 'quiz-option-wrong', 'quiz-option-selected');
            });
            explanationWrapper.classList.remove('quiz-explanation-visible');
            retryBtn.style.display = 'none';
        });

        // Reset savedAnswers and clear localStorage
        savedAnswers = {};
        try {
            localStorage.removeItem(storageKey);
        } catch (e) {
            console.error('Failed to remove quiz progress from localStorage', e);
        }

        updateScoreboard();
        scoreboard.scrollIntoView({ behavior: 'smooth' });
    });

    // Initial render
    updateScoreboard();
}



async function loadFile(filePath) {

    currentFilePath = filePath;

    document.getElementById('home-view').style.display = 'none';
    document.getElementById('reader-view').style.display = 'none';
    document.getElementById('search-view').style.display = 'none';
    document.getElementById('error-view').style.display = 'none';
    document.getElementById('author-view').style.display = 'none';
    document.getElementById('sidebar-tree-container').style.display = 'block';

    // Reset About the Author link styles
    document.getElementById('about-author-link').className = 'flex items-center gap-md p-3 text-on-surface-variant hover:bg-surface-container-high rounded-lg transition-colors';

    // Auto switch category matching the loading file path
    const matchedCat = categories.find(c => filePath.startsWith(c.dir));
    if (matchedCat) {
        selectCategory(matchedCat.name);
        highlightActiveFileInSidebar(filePath);
    }

    // Set up interactive breadcrumbs
    const breadcrumbsContainer = document.getElementById('breadcrumbs');
    breadcrumbsContainer.innerHTML = '';

    const homeSpan = document.createElement('span');
    homeSpan.innerText = 'Home';
    homeSpan.className = 'cursor-pointer hover:text-primary transition-colors';
    homeSpan.addEventListener('click', goHome);
    breadcrumbsContainer.appendChild(homeSpan);

    const parts = filePath.split('/');
    let accumulatedPath = '';

    parts.forEach((part, index) => {
        const sep = document.createElement('span');
        sep.className = 'text-on-surface-variant/40';
        sep.innerText = '/';
        breadcrumbsContainer.appendChild(sep);

        accumulatedPath = accumulatedPath ? accumulatedPath + '/' + part : part;
        const pathClosure = accumulatedPath;

        const span = document.createElement('span');
        const cleanPart = part.endsWith('.md') ? part.substring(0, part.length - 3) : part;
        span.innerText = cleanPart.split(/[-_]+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

        if (index === parts.length - 1) {
            span.className = 'text-on-surface font-semibold';
        } else {
            span.className = 'cursor-pointer hover:text-primary transition-colors';
            span.addEventListener('click', () => {
                navigateToResolvedPath(pathClosure);
            });
        }
        breadcrumbsContainer.appendChild(span);
    });

    try {
        const response = await fetch(filePath);
        if (!response.ok) throw new Error('File not found');
        let markdown = await response.text();

        markdown = parseAlertBlocks(markdown);

        // Protect LaTeX math blocks from being mangled by the markdown parser.
        // marked.parse() strips/corrupts backslashes (\begin, \boxed, \downarrow, etc.),
        // so we extract math blocks first, replace with unique placeholders, parse the
        // markdown, then restore the original LaTeX for KaTeX to render correctly.
        const mathPlaceholders = [];
        // Match display math ($$...$$) first, then inline math ($...$)
        markdown = markdown.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
            const id = `%%MATH_BLOCK_${mathPlaceholders.length}%%`;
            mathPlaceholders.push(match);
            return id;
        });
        markdown = markdown.replace(/(?<!\$)\$(?!\$)(.+?)\$(?!\$)/g, (match) => {
            const id = `%%MATH_BLOCK_${mathPlaceholders.length}%%`;
            mathPlaceholders.push(match);
            return id;
        });

        const markdownContentDiv = document.getElementById('markdown-content');
        let parsedHtml = marked.parse(markdown);

        // Restore the original LaTeX blocks from placeholders
        mathPlaceholders.forEach((original, index) => {
            parsedHtml = parsedHtml.replace(`%%MATH_BLOCK_${index}%%`, original);
        });

        markdownContentDiv.innerHTML = parsedHtml;

        // Convert mermaid code blocks into renderable <pre class="mermaid"> elements
        // marked outputs: <pre><code class="language-mermaid">...diagram code...</code></pre>
        // mermaid needs: <pre class="mermaid">...diagram code...</pre>
        markdownContentDiv.querySelectorAll('pre > code.language-mermaid').forEach(codeEl => {
            const pre = codeEl.parentElement;
            const diagramSource = codeEl.textContent;
            const mermaidPre = document.createElement('pre');
            mermaidPre.className = 'mermaid';
            mermaidPre.textContent = diagramSource;
            pre.replaceWith(mermaidPre);
        });

        Prism.highlightAllUnder(markdownContentDiv);

        // Render Mermaid Diagrams
        const mermaidNodes = markdownContentDiv.querySelectorAll('pre.mermaid');
        if (window.mermaid && mermaidNodes.length > 0) {
            mermaid.run({ nodes: mermaidNodes })
                .catch(err => console.error('Mermaid rendering error:', err));
        }

        addCopyButtons();
        fixRelativeImages();

        // Wait for mermaid to finish rendering, then init lightbox
        setTimeout(() => { initZoomLightbox(); }, 500);

        // Math LaTeX Auto-Render
        renderMathInElement(markdownContentDiv, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true }
            ],
            throwOnError: false
        });

        buildTOC();
        makeQuizInteractive();
        updateReaderDoneBtn();

        document.getElementById('reader-view').style.display = 'grid';
        window.scrollTo({ top: 0 });
        document.getElementById('progress-bar').style.width = '0%';
    } catch (err) {
        console.error(`Error loading markdown file ${filePath}:`, err);
        document.getElementById('error-view').style.display = 'flex';
    }
}

// Table of Contents Builder and Scrollspy
function buildTOC() {
    const container = document.getElementById('markdown-content');
    const tocNav = document.getElementById('toc-nav');
    tocNav.innerHTML = '';

    const headings = Array.from(container.querySelectorAll('h2, h3'))
        .filter(h => !h.closest('.readme-hero'));

    const asideElement = document.querySelector('aside.hidden.xl\\:block');
    if (headings.length === 0) {
        if (asideElement) asideElement.style.display = 'none';
        return;
    } else {
        if (asideElement) asideElement.style.display = 'block';
    }

    const seenSlugs = {};
    headings.forEach((heading, index) => {
        let slugId = heading.innerText
            .toLowerCase()
            .trim()
            .replace(/\s+/g, '-')
            .replace(/[^\w\-]+/g, '')
            .replace(/\-\-+/g, '-')
            .replace(/^-+/, '')
            .replace(/-+$/, '');

        if (slugId) {
            if (seenSlugs[slugId]) {
                seenSlugs[slugId]++;
                slugId = `${slugId}-${seenSlugs[slugId]}`;
            } else {
                seenSlugs[slugId] = 1;
            }
        }

        const id = heading.getAttribute('id') || slugId || `toc-heading-${index}`;
        heading.setAttribute('id', id);

        const a = document.createElement('a');
        a.href = `#${id}`;
        a.innerText = heading.innerText;

        if (heading.tagName.toLowerCase() === 'h3') {
            a.className = "toc-link block text-xs text-on-surface-variant hover:text-primary transition-colors pl-md py-1 border-l border-outline-variant";
        } else {
            a.className = "toc-link block text-sm text-on-surface-variant hover:text-primary transition-colors py-1";
        }

        a.addEventListener('click', (e) => {
            e.preventDefault();
            heading.scrollIntoView({ behavior: 'smooth' });
        });

        tocNav.appendChild(a);
    });

    if (window.tocObserver) {
        window.tocObserver.disconnect();
    }

    window.tocObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                document.querySelectorAll('.toc-link').forEach(link => {
                    link.classList.remove('text-primary', 'font-bold');
                    link.classList.add('text-on-surface-variant');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('text-primary', 'font-bold');
                        link.classList.remove('text-on-surface-variant');
                        link.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                });
            }
        });
    }, {
        root: null,
        rootMargin: '-80px 0px -60% 0px',
        threshold: 0.1
    });

    headings.forEach(h => window.tocObserver.observe(h));
}

// Live Search Handler
function handleSearch(query) {
    query = query.trim().toLowerCase();

    const homeView = document.getElementById('home-view');
    const readerView = document.getElementById('reader-view');
    const searchView = document.getElementById('search-view');
    const errorView = document.getElementById('error-view');
    const authorView = document.getElementById('author-view');

    if (!query) {
        if (currentFilePath === 'README.md') {
            homeView.style.display = 'block';
            readerView.style.display = 'none';
            authorView.style.display = 'none';
        } else if (window.location.hash === '#about') {
            homeView.style.display = 'none';
            readerView.style.display = 'none';
            authorView.style.display = 'block';
        } else {
            homeView.style.display = 'none';
            readerView.style.display = 'grid';
            authorView.style.display = 'none';
        }
        searchView.style.display = 'none';
        errorView.style.display = 'none';
        return;
    }

    homeView.style.display = 'none';
    readerView.style.display = 'none';
    errorView.style.display = 'none';
    authorView.style.display = 'none';
    searchView.style.display = 'block';

    const results = flattenedFiles.filter(item => {
        return item.name.toLowerCase().includes(query) ||
            item.path.toLowerCase().includes(query);
    });

    document.getElementById('search-results-count').innerText = `Found ${results.length} result(s) matching "${query}"`;

    const resultsList = document.getElementById('search-results-list');
    resultsList.innerHTML = '';

    if (results.length === 0) {
        resultsList.innerHTML = `
                    <div class="p-xl text-center text-on-surface-variant text-sm bg-surface border border-outline-variant rounded-xl">
                        No resources matched your search query. Try another keyword.
                    </div>
                `;
        return;
    }

    results.forEach(result => {
        const item = document.createElement('div');
        item.className = 'bg-surface border border-outline-variant p-lg rounded-xl hover:shadow-md hover:border-primary transition-all group flex flex-col gap-sm cursor-pointer';
        const folderPath = result.path.substring(0, result.path.lastIndexOf('/')) || 'Root';

        item.innerHTML = `
                    <h3 class="font-display text-xl text-on-surface group-hover:text-primary transition-colors">${result.name}</h3>
                    <div class="flex items-center gap-2 text-xs font-label text-on-surface-variant uppercase tracking-wider">
                        <span class="material-symbols-outlined text-sm">folder</span>
                        <span>${folderPath}</span>
                    </div>
                `;

        item.addEventListener('click', () => {
            window.location.hash = encodeURIComponent(result.path);
            document.getElementById('search-input').value = '';
        });

        resultsList.appendChild(item);
    });
}
// ────────────────────────────────────────
// Custom Cursor — home page only
// ────────────────────────────────────────
(function () {
    const dot = document.getElementById('cursor-dot');
    const ring = document.getElementById('cursor-ring');
    if (!dot || !ring) return;

    let mouseX = 0, mouseY = 0;
    let ringX = 0, ringY = 0;
    let raf;

    function lerp(a, b, t) { return a + (b - a) * t; }

    function tick() {
        // Dot snaps instantly
        dot.style.transform = `translate(calc(${mouseX}px - 50%), calc(${mouseY}px - 50%))`;
        // Ring trails behind
        ringX = lerp(ringX, mouseX, 0.12);
        ringY = lerp(ringY, mouseY, 0.12);
        ring.style.transform = `translate(calc(${ringX}px - 50%), calc(${ringY}px - 50%))`;
        raf = requestAnimationFrame(tick);
    }

    function isCursorViewActive() {
        const hv = document.getElementById('home-view');
        const av = document.getElementById('author-view');
        const isHome = hv && hv.style.display !== 'none';
        const isAuthor = av && av.style.display !== 'none';
        return isHome || isAuthor;
    }

    function enableCursor() {
        document.body.classList.add('home-cursor-active');
        dot.style.opacity = '1';
        ring.style.opacity = '1';
        if (!raf) raf = requestAnimationFrame(tick);
    }

    function disableCursor() {
        document.body.classList.remove('home-cursor-active');
        dot.style.opacity = '0';
        ring.style.opacity = '0';
        if (raf) { cancelAnimationFrame(raf); raf = null; }
    }

    // Track mouse
    document.addEventListener('mousemove', e => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        if (isCursorViewActive()) enableCursor();
        else disableCursor();
    });

    // Expand ring on interactive elements
    const hoverTargets = 'button, a, [onclick], .roadmap-row, .hero-cta-primary';
    document.addEventListener('mouseover', e => {
        if (e.target.closest(hoverTargets) && isCursorViewActive()) {
            ring.classList.add('expanded');
        }
    });
    document.addEventListener('mouseout', e => {
        if (e.target.closest(hoverTargets)) {
            ring.classList.remove('expanded');
        }
    });

    // Watch for view changes (hash navigation)
    window.addEventListener('hashchange', () => {
        if (!isCursorViewActive()) disableCursor();
    });

    // Initial state — hidden until mouse moves
    dot.style.opacity = '0';
    ring.style.opacity = '0';
})();
