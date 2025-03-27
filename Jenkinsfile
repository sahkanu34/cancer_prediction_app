pipeline {
    agent any

    environment {
        // Docker configuration
        DOCKER_IMAGE = 'sahkanu37/cancer_prediction'
        DOCKER_TAG = "${env.BUILD_ID}"

        // Kubernetes configuration
        K8S_NAMESPACE = 'cancer-prediction'
        MINIKUBE_PROFILE = 'minikube'  // Change if using a custom profile
        DEPLOYMENT_FILE = 'deployment.yaml'
        SERVICE_FILE = 'service.yaml'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    if (isUnix()) {
                        sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                    } else {
                        bat "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                    }
                }
            }
        }

        stage('Load into Minikube') {
            steps {
                script {
                    // Minikube must use its own Docker daemon
                    if (isUnix()) {
                        sh 'eval $(minikube docker-env)'
                        sh "docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest"
                    } else {
                        bat 'minikube docker-env | Invoke-Expression'
                        bat "docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest"
                    }
                }
            }
        }

        stage('Deploy to Minikube') {
            steps {
                script {
                    // Create namespace if not exists
                    if (isUnix()) {
                        sh """
                            minikube kubectl -- get ns ${K8S_NAMESPACE} || \
                            minikube kubectl -- create ns ${K8S_NAMESPACE}
                        """
                    } else {
                        bat """
                            minikube kubectl -- get ns ${K8S_NAMESPACE} || ^
                            minikube kubectl -- create ns ${K8S_NAMESPACE}
                        """
                    }

                    // Update image in deployment
                    if (isUnix()) {
                        sh """
                            sed -i 's|image: .*|image: ${DOCKER_IMAGE}:${DOCKER_TAG}|g' ${DEPLOYMENT_FILE}
                        """
                    } else {
                        bat """
                            powershell -Command "(Get-Content ${DEPLOYMENT_FILE}) -replace 'image: .*', 'image: ${DOCKER_IMAGE}:${DOCKER_TAG}' | Set-Content ${DEPLOYMENT_FILE}"
                        """
                    }

                    // Apply Kubernetes manifests
                    if (isUnix()) {
                        sh """
                            minikube kubectl -- -n ${K8S_NAMESPACE} apply -f ${DEPLOYMENT_FILE}
                            minikube kubectl -- -n ${K8S_NAMESPACE} apply -f ${SERVICE_FILE}
                        """
                    } else {
                        bat """
                            minikube kubectl -- -n ${K8S_NAMESPACE} apply -f ${DEPLOYMENT_FILE}
                            minikube kubectl -- -n ${K8S_NAMESPACE} apply -f ${SERVICE_FILE}
                        """
                    }

                    // Verify deployment
                    if (isUnix()) {
                        sh """
                            minikube kubectl -- -n ${K8S_NAMESPACE} rollout status deployment/cancer-prediction-app
                        """
                    } else {
                        bat """
                            minikube kubectl -- -n ${K8S_NAMESPACE} rollout status deployment/cancer-prediction-app
                        """
                    }
                }
            }
        }

        stage('Get Application URL') {
            steps {
                script {
                    if (isUnix()) {
                        sh """
                            echo "Application deployed to:"
                            minikube service -n ${K8S_NAMESPACE} cancer-prediction-service --url
                        """
                    } else {
                        bat """
                            echo "Application deployed to:"
                            minikube service -n ${K8S_NAMESPACE} cancer-prediction-service --url
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
                        docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true
                    """
                } else {
                    bat """
                        docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || echo "Image not found"
                    """
                }
            }
        }
    }
}