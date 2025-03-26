pipeline {
    agent any
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds')
        DOCKER_IMAGE = 'sahkanu37/cancer_prediction'
        K8S_NAMESPACE = 'cancer-prediction'
    }
    
    stages {
        stage('Checkout SCM') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${env.BUILD_ID}")
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                script {
                    // Login to Docker Hub
                    sh "docker login -u sahkanu37 -p ${DOCKERHUB_CREDENTIALS_PSW}"
                    
                    // Build and push image
                    sh "docker build -t ${DOCKER_IMAGE}:${env.BUILD_ID} ."
                    sh "docker push ${DOCKER_IMAGE}:${env.BUILD_ID}"
                    sh "docker tag ${DOCKER_IMAGE}:${env.BUILD_ID} ${DOCKER_IMAGE}:latest"
                    sh "docker push ${DOCKER_IMAGE}:latest"
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    // Update deployment image
                    sh """
                        sed -i 's|image: .*|image: ${DOCKER_IMAGE}:${env.BUILD_ID}|g' k8s/deployment.yaml
                        kubectl apply -f k8s/deployment.yaml -n ${K8S_NAMESPACE}
                        kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}
                        kubectl rollout status deployment/cancer-prediction -n ${K8S_NAMESPACE}
                    """
                }
            }
        }
    }
    
    post {
        always {
            // Clean up Docker images
            sh """
                docker rmi ${DOCKER_IMAGE}:${env.BUILD_ID} || true
                docker rmi ${DOCKER_IMAGE}:latest || true
            """
        }
        
        failure {
            echo 'Pipeline failed! Check Docker Hub credentials and Kubernetes configuration.'
        }
        
        success {
            echo 'Pipeline completed successfully!'
        }
    }
}
