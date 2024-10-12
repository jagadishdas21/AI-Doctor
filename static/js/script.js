// Function to handle user input and send message to the server
function sendMessage() {
    var userInput = document.getElementById("userInput").value;
    if (userInput) {
        displayMessage(userInput, 'user-msg');
        
        // Fetching the response from the Flask backend
        fetch(`/get?msg=${encodeURIComponent(userInput)}`)
            .then(response => response.json())  // Parse JSON response
            .then(data => displayMessage(data.response, 'bot-msg'))  // Use the response in your message display
            .catch(error => console.error('Error:', error)); // Handle errors
        
        document.getElementById("userInput").value = ""; // Clear input
    }
}


// Function to display a message in the chat window
function displayMessage(message, className) {
    const chatMessages = document.getElementById("chatMessages");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${className}`;
    messageDiv.innerText = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the latest message
}

// Function to handle suggested questions and send them automatically
function sendSuggestion(suggestion) {
    document.getElementById("userInput").value = suggestion; // Set the suggested question in the input field
    sendMessage(); // Send the message automatically when a suggestion is clicked
}