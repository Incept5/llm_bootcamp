
# UI for Local LLMs - Open WebUI Setup Guide

This guide explains how to set up Open WebUI to interact with your local Ollama LLMs, providing a user-friendly web interface similar to ChatGPT.

## What is Open WebUI?

Open WebUI (formerly Ollama WebUI) is a feature-rich and user-friendly self-hosted WebUI designed to operate entirely offline. It supports various LLM runners, including Ollama, and provides:

- 📱 Responsive web interface
- 🔒 Complete offline operation
- 👥 Multi-user support with role-based access
- 💬 Chat history and conversation management
- 🎨 Customizable interface themes
- 📁 Document upload and RAG support
- 🔌 Plugin system and integrations

## Prerequisites

Before setting up Open WebUI, ensure you have:

1. **Ollama installed and running**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   ```

2. **Docker installed** (recommended method)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows/Mac
   - Docker Engine for Linux

3. **At least one LLM model downloaded**
   ```bash
   # Download popular models
   ollama pull llama3.2
   ollama pull mistral
   ollama pull codellama
   ```

## Installation Methods

### Method 1: Docker (Recommended)

#### Quick Start with Docker Run
```bash
# Run Open WebUI with Ollama integration
docker run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

#### Using Docker Compose
Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      # Optional: Enable authentication
      - WEBUI_AUTH=False
      # Optional: Set admin user
      - WEBUI_SECRET_KEY=your-secret-key-here
    volumes:
      - open-webui:/app/backend/data
    restart: unless-stopped

volumes:
  open-webui:
```

Start with Docker Compose:
```bash
docker-compose up -d
```

### Method 2: Manual Installation

#### Using pip
```bash
# Install Open WebUI
pip install open-webui

# Start the application
open-webui serve --host 0.0.0.0 --port 3000
```

#### From Source
```bash
# Clone the repository
git clone https://github.com/open-webui/open-webui.git
cd open-webui

# Install dependencies
npm install
pip install -r backend/requirements.txt

# Build frontend
npm run build

# Start the application
cd backend
python main.py
```

## Configuration

### Environment Variables

Common configuration options:

```bash
# Ollama connection
OLLAMA_BASE_URL=http://localhost:11434

# Authentication
WEBUI_AUTH=True                    # Enable/disable authentication
WEBUI_SECRET_KEY=your-secret-key   # Secret key for sessions

# File uploads
ENABLE_RAG_WEB_SEARCH=True        # Enable web search for RAG
ENABLE_RAG_LOCAL_WEB_FETCH=True   # Enable local web fetching

# Model settings
DEFAULT_MODELS=llama3.2,mistral   # Default models to show
DEFAULT_USER_ROLE=user            # Default role for new users

# Interface customization
WEBUI_NAME="My Local AI"          # Custom interface name
```

### Connecting to Ollama

If Ollama is running on a different host or port:

```bash
# For remote Ollama instance
OLLAMA_BASE_URL=http://192.168.1.100:11434

# For Ollama with custom port
OLLAMA_BASE_URL=http://localhost:11435
```

## Usage

### First Time Setup

1. **Access the interface**: Open your browser and go to `http://localhost:3000`

2. **Create admin account**: On first visit, you'll be prompted to create an admin account

3. **Verify Ollama connection**: 
   - Go to Settings → Connections
   - Verify Ollama URL is correct
   - Test the connection

4. **Select models**: Available Ollama models should appear in the model dropdown

### Key Features

#### Chat Interface
- Select models from the dropdown
- Start conversations with your local LLMs
- Switch between different models mid-conversation
- Save and organize chat history

#### Document Upload and RAG
- Upload documents (PDF, TXT, DOCX, etc.)
- Ask questions about uploaded content
- Enable web search integration for enhanced responses

#### Multi-User Support
- Create multiple user accounts
- Set different permission levels
- Manage user access to specific models

#### Model Management
- View available Ollama models
- Pull new models directly from the interface
- Monitor model usage and performance

## Troubleshooting

### Common Issues

#### Can't Connect to Ollama
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If using Docker, check host connectivity
docker exec open-webui curl http://host.docker.internal:11434/api/tags
```

#### Models Not Appearing
```bash
# Verify models are installed
ollama list

# Pull models if missing
ollama pull llama3.2
```

#### Docker Host Connectivity (Linux)
On Linux, replace `host.docker.internal` with `172.17.0.1` or use `--network=host`:

```bash
docker run -d --network=host \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R 1000:1000 /path/to/open-webui/data
```

### Performance Optimization

#### For Better Performance
- Allocate sufficient RAM to Docker
- Use SSD storage for model files
- Consider GPU acceleration for Ollama
- Adjust model context length based on available memory

#### Resource Monitoring
```bash
# Monitor Docker container resources
docker stats open-webui

# Check Ollama resource usage
ollama ps
```

## Advanced Configuration

### Custom Models Configuration

Create a `models.json` file for custom model settings:

```json
{
  "models": [
    {
      "id": "llama3.2:8b",
      "name": "Llama 3.2 8B",
      "description": "Meta's Llama 3.2 8B parameter model",
      "size": "4.7GB",
      "capabilities": ["chat", "code"]
    }
  ]
}
```

### Nginx Reverse Proxy

For production deployment with SSL:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Security Considerations

### Authentication
- Always enable authentication in production: `WEBUI_AUTH=True`
- Use strong secret keys
- Regularly update admin passwords
- Consider OAuth integration for enterprise use

### Network Security
- Run behind a reverse proxy with SSL
- Use firewall rules to restrict access
- Consider VPN access for remote users
- Regular security updates

### Data Privacy
- All data stays local (offline operation)
- Chat history stored in local volumes
- No external API calls for core functionality
- Document uploads processed locally

## Integration Examples

### API Access
Open WebUI provides API endpoints compatible with OpenAI format:

```python
import requests

# Chat completion
response = requests.post('http://localhost:3000/api/chat', 
    json={
        'model': 'llama3.2',
        'messages': [{'role': 'user', 'content': 'Hello!'}]
    },
    headers={'Authorization': 'Bearer your-api-key'}
)
```

### Webhook Integration
Configure webhooks for external integrations:

```bash
# Environment variable
WEBHOOK_URL=http://your-webhook-endpoint.com/chat
```

## Useful Commands

```bash
# Start/stop services
docker start open-webui
docker stop open-webui

# View logs
docker logs open-webui -f

# Update to latest version
docker pull ghcr.io/open-webui/open-webui:main
docker stop open-webui
docker rm open-webui
# Then run the docker run command again

# Backup data
docker cp open-webui:/app/backend/data ./backup

# Clean up
docker system prune
docker volume prune
```

## Resources

- [Open WebUI GitHub](https://github.com/open-webui/open-webui)
- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Ollama Models](https://ollama.ai/library)
- [Docker Documentation](https://docs.docker.com/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/open-webui/open-webui/issues
- Discord Community: https://discord.gg/5rJgQTnV4s
- Documentation: https://docs.openwebui.com/

---

**Note**: This setup provides a complete local AI chat interface without sending any data to external services. All processing happens on your local machine, ensuring privacy and control over your AI interactions.
