pipeline {
    agent any
    
    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds')
        DOCKER_IMAGE = 'sahkanu37/cancer_prediction'
        K8S_NAMESPACE = 'cancer-prediction'
    }
    
    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${env.BUILD_ID}")
                }
            }
        }
        
        stage('Push to Docker Hub') {
            steps {
            withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                sh """
                docker login -u ${DOCKER_USER} -p ${DOCKER_PASS}
                docker push ${DOCKER_IMAGE}:${env.BUILD_ID}
                docker tag ${DOCKER_IMAGE}:${env.BUILD_ID} ${DOCKER_IMAGE}:latest
                docker push ${DOCKER_IMAGE}:latest
                """
            }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    sh "kubectl apply -f k8s/deployment.yaml -n ${K8S_NAMESPACE}"
                    sh "kubectl apply -f k8s/service.yaml -n ${K8S_NAMESPACE}"
                    sh "kubectl rollout status deployment/cancer-prediction -n ${K8S_NAMESPACE}"
                }
            }
        }
    }
    
    post {
        failure {
            echo 'Pipeline failed! Check Docker Hub credentials and Kubernetes configuration.'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
    }
}