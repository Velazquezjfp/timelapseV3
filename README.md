Missing model files and env files. 

This solution is a fine tuned detection model for vehicles and people

# Install gdown if not already installed
pip install gdown

# Download the file
gdown --id {insert ID} -O downloaded_file

# Try to unzip (for .zip), else fall back to tar (for .tar.gz, .tgz, etc.)
if file downloaded_file | grep -q "Zip archive"; then
    unzip downloaded_file -d extracted_contents
elif file downloaded_file | grep -q "gzip compressed"; then
    tar -xvzf downloaded_file -C extracted_contents
elif file downloaded_file | grep -q "bzip2 compressed"; then
    tar -xvjf downloaded_file -C extracted_contents
else
    echo "Unknown archive format, please check manually."
fi
