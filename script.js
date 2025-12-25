// Chat functionality with Flask API integration
class LegalChatBot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.userInput = document.getElementById('userInput');
        this.sendButton = document.getElementById('sendButton');
        this.apiUrl = 'http://localhost:5000/api/chat'; // Flask API endpoint
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = Math.min(this.userInput.scrollHeight, 120) + 'px';
        });
        
        // Check API health on load
        this.checkAPIHealth();
    }
    
    async checkAPIHealth() {
        try {
            const response = await fetch('http://127.0.0.1:5000/api/health');
            const data = await response.json();
            console.log('API Health:', data);
        } catch (error) {
            console.warn('API not available, using fallback responses');
        }
    }
    
    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.userInput.value = '';
        this.userInput.style.height = 'auto';
        
        // Disable send button
        this.sendButton.disabled = true;
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Get bot response
        try {
            const response = await this.getBotResponse(message);
            this.hideTypingIndicator();
            this.addMessage(response, 'assistant');
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
            console.error('Error:', error);
        }
        
        // Re-enable send button
        this.sendButton.disabled = false;
    }
    
    async getBotResponse(message) {
        try {
            // Try to call Flask API first
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (response.ok) {
                const data = await response.json();
                return data.response;
            } else {
                throw new Error('API request failed');
            }
        } catch (error) {
            console.log('API not available, using fallback responses');
            // Fallback to local responses if API is not available
            return this.getFallbackResponse(message);
        }
    }
    
    getFallbackResponse(message) {
        // Local fallback responses when API is not available
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('contract')) {
            return "A contract is a legally binding agreement between two or more parties. Key elements include offer, acceptance, consideration, and mutual intent. Always ensure contracts are in writing for important matters and consider having them reviewed by a licensed attorney.\n\n*Please note: This is general information only. Consult a licensed attorney for specific legal advice.*";
        } else if (lowerMessage.includes('divorce')) {
            return "Divorce laws vary by jurisdiction. Common considerations include property division, child custody, spousal support, and filing procedures. I strongly recommend consulting with a family law attorney who can provide guidance specific to your situation and local laws.\n\n*Please note: This is general information only. Consult a licensed attorney for specific legal advice.*";
        } else if (lowerMessage.includes('tenant') || lowerMessage.includes('landlord')) {
            return "Tenant-landlord laws cover rights and responsibilities of both parties. Key areas include lease agreements, security deposits, maintenance obligations, and eviction procedures. These laws vary significantly by location. Please consult local housing authorities or a real estate attorney for specific guidance.\n\n*Please note: This is general information only. Consult a licensed attorney for specific legal advice.*";
        } else if (lowerMessage.includes('employment') || lowerMessage.includes('workplace')) {
            return "Employment law covers workplace rights, discrimination, wages, and termination. Important areas include at-will employment, protected classes, overtime regulations, and workplace safety. For specific employment issues, consider contacting your local labor department or an employment attorney.\n\n*Please note: This is general information only. Consult a licensed attorney for specific legal advice.*";
        } else if (lowerMessage.includes('criminal') || lowerMessage.includes('arrest')) {
            return "Criminal law matters are serious. If you're facing criminal charges or have been arrested, you have the right to remain silent and the right to an attorney. I strongly recommend seeking immediate legal representation from a criminal defense attorney.\n\n*Please note: This is general information only. Consult a licensed attorney for specific legal advice.*";
        } else if (lowerMessage.includes('will') || lowerMessage.includes('estate')) {
            return "Estate planning involves preparing for the transfer of assets after death. Key documents include wills, trusts, powers of attorney, and healthcare directives. Consider consulting with an estate planning attorney to ensure your wishes are properly documented.\n\n*Please note: This is general information only. Consult a licensed attorney for specific legal advice.*";
        } else {
            return "I can provide general information about various legal topics including contracts, family law, tenant rights, employment law, criminal law, and estate planning. However, this information is educational only and not a substitute for professional legal advice. What specific legal topic would you like to know more about?\n\n*For specific legal matters, please consult with a licensed attorney in your jurisdiction.*";
        }
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Parse content for better formatting
        const formattedContent = content.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
        
        // Check if content has disclaimer (italic text)
        if (content.includes('*') && content.includes('*')) {
            contentDiv.innerHTML = formattedContent.replace(/\*(.*?)\*/g, '<em style="font-size: 12px; opacity: 0.8;">$1</em>');
        } else {
            contentDiv.innerHTML = formattedContent;
        }
        
        messageDiv.appendChild(contentDiv);
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message';
        typingDiv.id = 'typingIndicator';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        
        typingDiv.appendChild(indicator);
        this.chatMessages.appendChild(typingDiv);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LegalChatBot();
});