pipeline {
    agent any
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds')
        DOCKER_IMAGE = 'sahkanu37/cancer_prediction'
        K8S_NAMESPACE = 'cancer-prediction'
        DEPLOYMENT_PATH = 'E:/Streamlit-App-Cancer/deployment.yaml'
        SERVICE_PATH = 'E:/Streamlit-App-Cancer/service.yaml'
    }
    
    stages {
        stage('Validate Environment') {
            steps {
                script {
                    if (!fileExists(env.DEPLOYMENT_PATH)) {
                        error "Deployment file not found at ${env.DEPLOYMENT_PATH}"
                    }
                    if (!fileExists(env.SERVICE_PATH)) {
                        error "Service file not found at ${env.SERVICE_PATH}"
                    }
                }
            }
        }
        
        stage('Checkout SCM') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    try {
                        if (isUnix()) {
                            sh "docker version"
                            sh "docker build -t ${DOCKER_IMAGE}:${env.BUILD_ID} ."
                        } else {
                            bat "docker version"
                            bat "docker build -t ${DOCKER_IMAGE}:${env.BUILD_ID} ."
                        }
                    } catch (Exception e) {
                        error "Docker build failed: ${e.message}"
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
                        try {
                            if (isUnix()) {
                                sh """
                                    echo "${DOCKER_PASS}" | docker login -u ${DOCKER_USER} --password-stdin
                                    docker push ${DOCKER_IMAGE}:${env.BUILD_ID}
                                    docker tag ${DOCKER_IMAGE}:${env.BUILD_ID} ${DOCKER_IMAGE}:latest
                                    docker push ${DOCKER_IMAGE}:latest
                                """
                            } else {
                                bat """
                                    echo logging in to Docker Hub
                                    docker login -u ${DOCKER_USER} -p ${DOCKER_PASS}
                                    docker push ${DOCKER_IMAGE}:${env.BUILD_ID}
                                    docker tag ${DOCKER_IMAGE}:${env.BUILD_ID} ${DOCKER_IMAGE}:latest
                                    docker push ${DOCKER_IMAGE}:latest
                                """
                            }
                        } catch (Exception e) {
                            error "Docker push failed: ${e.message}"
                        }
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            when {
                expression { 
                    currentBuild.resultIsBetterOrEqualTo('SUCCESS') 
                }
            }
            steps {
                script {
                    try {
                        if (isUnix()) {
                            sh "kubectl config current-context"
                            sh "kubectl get namespace ${K8S_NAMESPACE} || kubectl create namespace ${K8S_NAMESPACE}"
                            
                            sh """
                                sed -i 's|image: .*|image: ${DOCKER_IMAGE}:${env.BUILD_ID}|g' "${env.DEPLOYMENT_PATH}"
                                kubectl apply -f "${env.DEPLOYMENT_PATH}" -n ${K8S_NAMESPACE}
                                kubectl apply -f "${env.SERVICE_PATH}" -n ${K8S_NAMESPACE}
                                kubectl rollout status deployment/cancer-prediction -n ${K8S_NAMESPACE} --timeout=300s
                            """
                        } else {
                            bat """
                                kubectl config current-context
                                kubectl get namespace ${K8S_NAMESPACE} || kubectl create namespace ${K8S_NAMESPACE}
                                
                                powershell -Command "(Get-Content '${env.DEPLOYMENT_PATH}') -replace 'image: .*', 'image: ${DOCKER_IMAGE}:${env.BUILD_ID}' | Set-Content '${env.DEPLOYMENT_PATH}'"
                                kubectl apply -f "${env.DEPLOYMENT_PATH}" -n ${K8S_NAMESPACE}
                                kubectl apply -f "${env.SERVICE_PATH}" -n ${K8S_NAMESPACE}
                                kubectl rollout status deployment/cancer-prediction -n ${K8S_NAMESPACE} --timeout=300s
                            """
                        }
                    } catch (Exception e) {
                       
                        error "Kubernetes deployment failed: ${e.message}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            script {
                try {
                    if (isUnix()) {
                        sh """
                            docker rmi ${DOCKER_IMAGE}:${env.BUILD_ID} || true
                            docker rmi ${DOCKER_IMAGE}:latest || true
                            docker logout
                        """
                    } else {
                        bat """
                            docker rmi ${DOCKER_IMAGE}:${env.BUILD_ID} || echo "Build image not found"
                            docker rmi ${DOCKER_IMAGE}:latest || echo "Latest image not found"
                            docker logout
                        """
                    }
                } catch (Exception e) {
                    echo "Cleanup encountered an error: ${e.message}"
                }
            }
        }
        
        failure {
            // Enhanced failure notification
            script {
                def failureMessage = """
                    Pipeline Failed! 
                    Job: ${env.JOB_NAME}
                    Build Number: ${env.BUILD_NUMBER}
                    Build URL: ${env.BUILD_URL}
                    
                    Possible issues:
                    - Docker build/push failed
                    - Kubernetes deployment configuration incorrect
                    - Credentials or authentication problems
                """
                
                echo failureMessage
            }
        }
        
        success {
            script {
                echo "Pipeline completed successfully!"
            }
        }
    }
}