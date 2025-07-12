"""
Deployment Module for Linear Regression Model
Handles Docker configuration, CI/CD pipeline, and production deployment
"""

import os
import subprocess
import logging
import json
import yaml
from typing import Dict, Any, Optional
import docker
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DockerDeployer:
    """Handle Docker-based deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DockerDeployer
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.client = docker.from_env()
    
    def create_dockerfile(self, output_path: str = "Dockerfile"):
        """Create Dockerfile for the application"""
        dockerfile_content = f"""
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/best_model.pkl
ENV PREPROCESSOR_PATH=/app/models/preprocessor.pkl

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Dockerfile created at {output_path}")
    
    def create_docker_compose(self, output_path: str = "docker-compose.yml"):
        """Create docker-compose.yml file"""
        compose_content = {
            "version": "3.8",
            "services": {
                "ml-api": {
                    "build": ".",
                    "ports": ["8000:8000"],
                    "environment": [
                        "MODEL_PATH=/app/models/best_model.pkl",
                        "PREPROCESSOR_PATH=/app/models/preprocessor.pkl"
                    ],
                    "volumes": [
                        "./models:/app/models",
                        "./logs:/app/logs"
                    ],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": "3",
                        "start_period": "40s"
                    }
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "ports": ["80:80"],
                    "volumes": ["./nginx.conf:/etc/nginx/nginx.conf"],
                    "depends_on": ["ml-api"],
                    "restart": "unless-stopped"
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
        
        logger.info(f"Docker Compose file created at {output_path}")
    
    def create_nginx_config(self, output_path: str = "nginx.conf"):
        """Create nginx configuration for load balancing"""
        nginx_content = """
events {
    worker_connections 1024;
}

http {
    upstream ml_api {
        server ml-api:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://ml_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
        
        with open(output_path, 'w') as f:
            f.write(nginx_content)
        
        logger.info(f"Nginx configuration created at {output_path}")
    
    def build_image(self, tag: str = "ml-api:latest"):
        """Build Docker image"""
        try:
            logger.info(f"Building Docker image: {tag}")
            image, logs = self.client.images.build(
                path=".",
                tag=tag,
                rm=True
            )
            logger.info(f"Docker image built successfully: {image.tags}")
            return image
        except Exception as e:
            logger.error(f"Error building Docker image: {str(e)}")
            raise
    
    def run_container(self, image_tag: str = "ml-api:latest", port: int = 8000):
        """Run Docker container"""
        try:
            logger.info(f"Running container from image: {image_tag}")
            container = self.client.containers.run(
                image_tag,
                ports={'8000/tcp': port},
                environment={
                    'MODEL_PATH': '/app/models/best_model.pkl',
                    'PREPROCESSOR_PATH': '/app/models/preprocessor.pkl'
                },
                volumes={
                    os.path.abspath('models'): {'bind': '/app/models', 'mode': 'ro'},
                    os.path.abspath('logs'): {'bind': '/app/logs', 'mode': 'rw'}
                },
                detach=True,
                name="ml-api-container"
            )
            logger.info(f"Container started: {container.id}")
            return container
        except Exception as e:
            logger.error(f"Error running container: {str(e)}")
            raise


class CICDPipeline:
    """Handle CI/CD pipeline configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CICDPipeline
        
        Args:
            config: CI/CD configuration
        """
        self.config = config
    
    def create_github_actions(self, output_path: str = ".github/workflows/deploy.yml"):
        """Create GitHub Actions workflow"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        workflow_content = {
            "name": "Deploy ML Model",
            "on": {
                "push": {
                    "branches": ["main"]
                },
                "pull_request": {
                    "branches": ["main"]
                }
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "uses": "actions/checkout@v2"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v2",
                            "with": {
                                "python-version": "3.9"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": "python -m pytest tests/ -v"
                        },
                        {
                            "name": "Run linting",
                            "run": "flake8 src/ tests/"
                        }
                    ]
                },
                "build-and-deploy": {
                    "runs-on": "ubuntu-latest",
                    "needs": "test",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {
                            "uses": "actions/checkout@v2"
                        },
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v1"
                        },
                        {
                            "name": "Login to Docker Hub",
                            "uses": "docker/login-action@v1",
                            "with": {
                                "username": "${{ secrets.DOCKER_USERNAME }}",
                                "password": "${{ secrets.DOCKER_PASSWORD }}"
                            }
                        },
                        {
                            "name": "Build and push Docker image",
                            "uses": "docker/build-push-action@v2",
                            "with": {
                                "context": ".",
                                "push": "true",
                                "tags": "${{ secrets.DOCKER_USERNAME }}/ml-api:latest"
                            }
                        },
                        {
                            "name": "Deploy to production",
                            "run": "echo 'Deploy to production server'"
                        }
                    ]
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(workflow_content, f, default_flow_style=False)
        
        logger.info(f"GitHub Actions workflow created at {output_path}")
    
    def create_jenkins_pipeline(self, output_path: str = "Jenkinsfile"):
        """Create Jenkins pipeline"""
        jenkins_content = """
pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'ml-api'
        DOCKER_TAG = 'latest'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Run Tests') {
            steps {
                sh 'python -m pytest tests/ -v'
            }
        }
        
        stage('Lint Code') {
            steps {
                sh 'flake8 src/ tests/'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .'
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh 'docker-compose -f docker-compose.staging.yml up -d'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                sh 'docker-compose -f docker-compose.prod.yml up -d'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
"""
        
        with open(output_path, 'w') as f:
            f.write(jenkins_content)
        
        logger.info(f"Jenkins pipeline created at {output_path}")


class ProductionDeployer:
    """Handle production deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ProductionDeployer
        
        Args:
            config: Production deployment configuration
        """
        self.config = config
    
    def create_production_compose(self, output_path: str = "docker-compose.prod.yml"):
        """Create production docker-compose file"""
        prod_compose = {
            "version": "3.8",
            "services": {
                "ml-api": {
                    "image": "ml-api:latest",
                    "ports": ["8000:8000"],
                    "environment": [
                        "MODEL_PATH=/app/models/best_model.pkl",
                        "PREPROCESSOR_PATH=/app/models/preprocessor.pkl",
                        "LOG_LEVEL=INFO"
                    ],
                    "volumes": [
                        "./models:/app/models:ro",
                        "./logs:/app/logs"
                    ],
                    "restart": "always",
                    "deploy": {
                        "replicas": 3,
                        "resources": {
                            "limits": {
                                "cpus": "1.0",
                                "memory": "1G"
                            }
                        }
                    },
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": "3",
                        "start_period": "40s"
                    }
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "ports": ["80:80", "443:443"],
                    "volumes": [
                        "./nginx.prod.conf:/etc/nginx/nginx.conf",
                        "./ssl:/etc/nginx/ssl"
                    ],
                    "depends_on": ["ml-api"],
                    "restart": "always"
                },
                "monitoring": {
                    "image": "prom/prometheus:latest",
                    "ports": ["9090:9090"],
                    "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml"],
                    "restart": "always"
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(prod_compose, f, default_flow_style=False)
        
        logger.info(f"Production docker-compose created at {output_path}")
    
    def create_monitoring_config(self, output_path: str = "prometheus.yml"):
        """Create Prometheus monitoring configuration"""
        prometheus_config = {
            "global": {
                "scrape_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "ml-api",
                    "static_configs": [
                        {
                            "targets": ["ml-api:8000"]
                        }
                    ],
                    "metrics_path": "/metrics"
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        logger.info(f"Prometheus configuration created at {output_path}")
    
    def deploy_to_production(self):
        """Deploy to production environment"""
        try:
            logger.info("Starting production deployment...")
            
            # Stop existing containers
            subprocess.run(["docker-compose", "-f", "docker-compose.prod.yml", "down"], 
                         check=True)
            
            # Pull latest images
            subprocess.run(["docker-compose", "-f", "docker-compose.prod.yml", "pull"], 
                         check=True)
            
            # Start new containers
            subprocess.run(["docker-compose", "-f", "docker-compose.prod.yml", "up", "-d"], 
                         check=True)
            
            logger.info("Production deployment completed successfully!")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise


def create_deployment_config() -> Dict[str, Any]:
    """Create default deployment configuration"""
    return {
        "docker": {
            "base_image": "python:3.9-slim",
            "port": 8000,
            "health_check_interval": "30s"
        },
        "ci_cd": {
            "platform": "github_actions",  # or "jenkins"
            "auto_deploy": True,
            "test_before_deploy": True
        },
        "production": {
            "replicas": 3,
            "load_balancer": True,
            "monitoring": True,
            "ssl": True
        }
    }


if __name__ == "__main__":
    # Example usage
    config = create_deployment_config()
    
    # Create Docker deployment files
    docker_deployer = DockerDeployer(config["docker"])
    docker_deployer.create_dockerfile()
    docker_deployer.create_docker_compose()
    docker_deployer.create_nginx_config()
    
    # Create CI/CD pipeline
    cicd = CICDPipeline(config["ci_cd"])
    cicd.create_github_actions()
    cicd.create_jenkins_pipeline()
    
    # Create production deployment
    prod_deployer = ProductionDeployer(config["production"])
    prod_deployer.create_production_compose()
    prod_deployer.create_monitoring_config()
    
    print("Deployment configuration files created successfully!") 