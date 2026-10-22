const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const pdfUpload = document.getElementById('pdf-upload');
const uploadStatus = document.getElementById('upload-status');
const clearChatBtn = document.getElementById('clear-chat');
const sampleQs = document.querySelectorAll('.sample-q');

// Handle sample questions
sampleQs.forEach(q => {
    q.addEventListener('click', () => {
        userInput.value = q.textContent;
        sendMessage();
    });
});

// Clear chat
clearChatBtn.addEventListener('click', () => {
    chatBox.innerHTML = `
        <div class="message bot-msg">
            <div class="message-content">Chat cleared. How can I help you?</div>
            <div class="message-time">Just now</div>
        </div>
    `;
});

// Handle PDF upload
pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    uploadStatus.innerHTML = '<span style="color: #38bdf8"><i class="fas fa-spinner fa-spin"></i> Uploading...</span>';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            uploadStatus.innerHTML = '<span style="color: #10b981"><i class="fas fa-check-circle"></i> Ingested successfully</span>';
        } else {
            uploadStatus.innerHTML = `<span style="color: #ef4444"><i class="fas fa-times-circle"></i> Error: ${data.message}</span>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = '<span style="color: #ef4444"><i class="fas fa-times-circle"></i> Upload failed</span>';
    }
});

// Send message on Enter
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

async function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    // Disable input while processing
    userInput.disabled = true;
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    // Add user message (the question)
    addMessage(question, 'user-msg');
    userInput.value = '';

    // Add typing indicator for the bot response
    const typingId = 'typing-' + Date.now();
    addTypingIndicator(typingId);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        removeTypingIndicator(typingId);
        
        // Add bot message (the answer)
        addMessage(data.response, 'bot-msg', true);
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage("I'm sorry, I encountered an error. Please try again later.", 'bot-msg');
    } finally {
        // Re-enable input
        userInput.disabled = false;
        userInput.focus();
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    }
}

function addMessage(text, className, isMarkdown = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isMarkdown) {
        contentDiv.innerHTML = marked.parse(text);
    } else {
        contentDiv.textContent = text;
    }
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    msgDiv.appendChild(contentDiv);
    msgDiv.appendChild(timeDiv);
    chatBox.appendChild(msgDiv);
    
    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addTypingIndicator(id) {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-msg typing-msg';
    typingDiv.id = id;
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}
