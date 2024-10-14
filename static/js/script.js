// JavaScript for handling chat interactions

function sendMessage() {
    const inputField = document.getElementById('userInput');
    const message = inputField.value.trim();
    if (message === '') return;

    // Add user message to chat
    addMessageToChat(message, 'user-msg');

    // Simulate a bot response (this can be replaced with real chatbot logic)
    setTimeout(() => {
        const botResponse = getBotResponse(message);
        addMessageToChat(botResponse, 'bot-msg');
    }, 500); // Simulate response delay

    inputField.value = ''; // Clear input field
}

function sendSuggestion(message) {
    // Send suggested message
    document.getElementById('userInput').value = message;
    sendMessage();
}

function addMessageToChat(message, className) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    messageDiv.textContent = message;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight; // Auto-scroll to bottom
}

function getBotResponse(message) {
    // Basic bot response for now (replace with your logic)
    if (message.toLowerCase().includes('hello')) {
        return 'Hello! How can I assist you today?';
    } else {
        return 'I’m here to help! Please ask a health-related question.';
    }
}