#!/usr/bin/env python3
"""
Deployment script for agentic-reviewer system.

This script sets up the system for production deployment with proper configuration,
security settings, and monitoring.
"""

import os
import sys
import subprocess
import secrets
import argparse
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "outputs",
        "logs",
        "backups",
        "ssl"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")


def generate_ssl_certificates():
    """Generate self-signed SSL certificates for HTTPS."""
    if not os.path.exists("ssl/cert.pem") or not os.path.exists("ssl/key.pem"):
        print("🔐 Generating SSL certificates...")
        
        # Generate private key
        run_command(
            "openssl genrsa -out ssl/key.pem 2048",
            "Generating private key"
        )
        
        # Generate certificate
        run_command(
            "openssl req -new -x509 -key ssl/key.pem -out ssl/cert.pem -days 365 -subj '/CN=localhost'",
            "Generating SSL certificate"
        )
        
        print("✅ SSL certificates generated")
    else:
        print("✅ SSL certificates already exist")


def setup_environment():
    """Set up environment variables for production."""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print("🔧 Creating environment configuration...")
        
        # Generate secure API key
        api_key = secrets.token_urlsafe(32)
        
        env_content = f"""# Production Environment Configuration
AR_MODEL_NAME=mistral
AR_OLLAMA_URL=http://localhost:11434
AR_TEMPERATURE=0.1
AR_MAX_TOKENS=512
AR_TIMEOUT=30

# API Configuration
AR_API_HOST=0.0.0.0
AR_API_PORT=8000
AR_API_KEY={api_key}
AR_RATE_LIMIT_MAX=1000

# Performance Configuration
AR_BATCH_SIZE=10
AR_MAX_CONCURRENT=10
AR_CACHE_MAX_SIZE_MB=200

# Security Configuration
AR_ENABLE_SANITIZATION=true

# Logging Configuration
AR_LOG_LEVEL=INFO
AR_DB_PATH=outputs/reviewed_predictions.sqlite
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Environment configuration created")
        print(f"🔑 API Key generated: {api_key[:16]}...")
        print("⚠️  Save this API key securely!")
    else:
        print("✅ Environment configuration already exists")


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install dependencies")
        return False
    
    return True


def setup_database():
    """Initialize the database."""
    print("🗄️  Setting up database...")
    
    try:
        from core.logger import AuditLogger
        logger = AuditLogger()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def setup_logging():
    """Set up logging configuration."""
    print("📝 Setting up logging...")
    
    log_config = """# Logging configuration
version: 1
formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
root:
  level: INFO
  handlers: [console, file]
"""
    
    with open("logging.conf", 'w') as f:
        f.write(log_config)
    
    print("✅ Logging configuration created")


def create_systemd_service():
    """Create systemd service file for production deployment."""
    service_content = """[Unit]
Description=Agentic Reviewer API
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/agentic-reviewer
Environment=PATH=/path/to/agentic-reviewer/.venv/bin
ExecStart=/path/to/agentic-reviewer/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("agentic-reviewer.service", 'w') as f:
        f.write(service_content)
    
    print("✅ Systemd service file created")
    print("⚠️  Update the paths in agentic-reviewer.service before installing")


def create_dockerfile():
    """Create Dockerfile for containerized deployment."""
    dockerfile_content = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p outputs logs backups

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]
"""
    
    with open("Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created")


def create_docker_compose():
    """Create docker-compose.yml for easy deployment."""
    compose_content = """version: '3.8'

services:
  agentic-reviewer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AR_MODEL_NAME=mistral
      - AR_OLLAMA_URL=http://ollama:11434
      - AR_API_KEY=${AR_API_KEY}
      - AR_RATE_LIMIT_MAX=1000
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    depends_on:
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
"""
    
    with open("docker-compose.yml", 'w') as f:
        f.write(compose_content)
    
    print("✅ Docker Compose file created")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    if run_command("python -m pytest tests/ -v", "Running test suite"):
        print("✅ All tests passed")
        return True
    else:
        print("❌ Some tests failed")
        return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy agentic-reviewer system")
    parser.add_argument("--mode", choices=["dev", "prod", "docker"], default="prod",
                       help="Deployment mode")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running tests")
    parser.add_argument("--ssl", action="store_true",
                       help="Generate SSL certificates")
    
    args = parser.parse_args()
    
    print("🚀 Starting deployment...")
    print(f"📋 Mode: {args.mode}")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Deployment failed: Could not install dependencies")
        sys.exit(1)
    
    # Run tests (unless skipped)
    if not args.skip_tests:
        if not run_tests():
            print("❌ Deployment failed: Tests failed")
            sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Setup database
    if not setup_database():
        print("❌ Deployment failed: Could not setup database")
        sys.exit(1)
    
    # Setup logging
    setup_logging()
    
    # Generate SSL certificates if requested
    if args.ssl:
        generate_ssl_certificates()
    
    # Create deployment files based on mode
    if args.mode == "prod":
        create_systemd_service()
        print("📋 Production deployment files created")
        print("📝 Next steps:")
        print("   1. Update paths in agentic-reviewer.service")
        print("   2. Copy agentic-reviewer.service to /etc/systemd/system/")
        print("   3. Run: sudo systemctl enable agentic-reviewer")
        print("   4. Run: sudo systemctl start agentic-reviewer")
    
    elif args.mode == "docker":
        create_dockerfile()
        create_docker_compose()
        print("📋 Docker deployment files created")
        print("📝 Next steps:")
        print("   1. Set AR_API_KEY environment variable")
        print("   2. Run: docker-compose up -d")
    
    else:  # dev mode
        print("📋 Development deployment completed")
        print("📝 Next steps:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Run: python main.py")
    
    print("✅ Deployment completed successfully!")


if __name__ == "__main__":
    main() 