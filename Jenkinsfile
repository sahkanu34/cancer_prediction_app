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
                    if (isUnix()) {
                        sh "docker build -t ${DOCKER_IMAGE}:${env.BUILD_ID} ."
                    } else {
                        bat "docker build -t ${DOCKER_IMAGE}:${env.BUILD_ID} ."
                    }
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    script {
                        if (isUnix()) {
                            sh """
                                docker login -u ${DOCKER_USER} -p ${DOCKER_PASS}
                                docker push ${DOCKER_IMAGE}:${env.BUILD_ID}
                                docker tag ${DOCKER_IMAGE}:${env.BUILD_ID} ${DOCKER_IMAGE}:latest
                                docker push ${DOCKER_IMAGE}:latest
                            """
                        } else {
                            bat """
                                docker login -u %DOCKER_USER% -p %DOCKER_PASS%
                                docker push ${DOCKER_IMAGE}:${env.BUILD_ID}
                                docker tag ${DOCKER_IMAGE}:${env.BUILD_ID} ${DOCKER_IMAGE}:latest
                                docker push ${DOCKER_IMAGE}:latest
                            """
                        }
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            when {
                expression { currentBuild.resultIsBetterOrEqualTo('SUCCESS') }
            }
            steps {
                script {
                    // Verify kubectl is available
                    if (isUnix()) {
                        sh """
                            Update deployment image
                            sed -i 's|image: .*|image: ${DOCKER_IMAGE}:${env.BUILD_ID}|g' k8s/deployment.yaml
                            
                            Apply Kubernetes manifests
                            kubectl apply -f k8s/deployment.yaml -n ${K8S_NAMESPACE}
                            kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}
                            
                            Verify deployment
                            kubectl rollout status deployment/cancer-prediction -n ${K8S_NAMESPACE} --timeout=300s
                        """
                    } else {
                        bat """
                            Update deployment image (Windows version)
                            powershell -Command "(Get-Content k8s/deployment.yaml) -replace 'image: .*', 'image: ${DOCKER_IMAGE}:${env.BUILD_ID}' | Set-Content k8s/deployment.yaml"
                            
                            Apply Kubernetes manifests
                            kubectl apply -f k8s/deployment.yaml -n ${K8S_NAMESPACE}
                            kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}
                            
                            Verify deployment
                            kubectl rollout status deployment/cancer-prediction -n ${K8S_NAMESPACE} --timeout=300s
                        """
                    }
                }
            }
        }
    }
    
    post {
        always {
            script {
                // Clean up Docker images
                if (isUnix()) {
                    sh """
                        docker rmi ${DOCKER_IMAGE}:${env.BUILD_ID} || true
                        docker rmi ${DOCKER_IMAGE}:latest || true
                    """
                } else {
                    bat """
                        docker rmi ${DOCKER_IMAGE}:${env.BUILD_ID} || echo "Image not found"
                        docker rmi ${DOCKER_IMAGE}:latest || echo "Image not found"
                    """
                }
            }
        }
        
        failure {
            echo 'Pipeline failed! Check Docker Hub credentials and Kubernetes configuration.'
            // Add notification steps here (email, Slack, etc.)
        }
        
        success {
            echo 'Pipeline completed successfully!'
            // Add success notification if needed
        }
    }
}