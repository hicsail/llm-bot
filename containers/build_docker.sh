#docker build -t myimage .
docker buildx build --platform linux/amd64 -t us-east1-docker.pkg.dev/special-michelle/naacp/gdp_api .

# We push the image to Michelle's special project
docker push us-east1-docker.pkg.dev/special-michelle/naacp/gdp_api                                    
