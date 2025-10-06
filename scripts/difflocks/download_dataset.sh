#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Get the download path from the first argument
TARGET_PATH="$1"

# Check if the path was given
if [ -z "$TARGET_PATH" ]; then
  echo "Usage: $0 <download-path>"
  exit 1
fi

# Make sure the directory exists
mkdir -p "$TARGET_PATH"

# username and password input
echo -e "\nYou need to register at https://difflocks.is.tue.mpg.de/"
read -p "Username (DIFFLOCKS):" username
read -p "Password (DIFFLOCKS):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading files..."


#BODY DATA
filename='difflocks_dataset_body_data.7z'
wget --post-data "username=$username&password=$password" \
  "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=$filename" \
  -O "$TARGET_PATH/$filename" --no-check-certificate --continue
# Check if wget failed (non-zero exit code)
if [ $? -ne 0 ]; then
    echo "❌ Error downloading body data. Exiting script."
    exit 1
fi

#IMGS
for i in $(seq 0 20); do
    filename="difflocks_dataset_imgs_chunk_${i}.7z"
     # Run wget and capture status
    wget --post-data "username=$username&password=$password" \
         "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=${filename}" \
         -O "$TARGET_PATH/$filename" --no-check-certificate --continue
    # Check if wget failed (non-zero exit code)
    if [ $? -ne 0 ]; then
        echo "❌ Error downloading $filename. Exiting script."
        exit 1
    fi
done

#IMGS v2
for i in $(seq 0 20); do
    filename="difflocks_dataset_imgs_v2_chunk_${i}.7z"
     # Run wget and capture status
    wget --post-data "username=$username&password=$password" \
         "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=${filename}" \
         -O "$TARGET_PATH/$filename" --no-check-certificate --continue
    # Check if wget failed (non-zero exit code)
    if [ $? -ne 0 ]; then
        echo "❌ Error downloading $filename. Exiting script."
        exit 1
    fi
done

#HAIRSTYLES
for i in $(seq 0 4455); do
    filename="difflocks_dataset_hairstyles_chunk_${i}.7z"
     # Run wget and capture status
    wget --post-data "username=$username&password=$password" \
         "https://download.is.tue.mpg.de/download.php?domain=difflocks&sfile=${filename}" \
         -O "$TARGET_PATH/$filename" --no-check-certificate --continue
    # Check if wget failed (non-zero exit code)
    if [ $? -ne 0 ]; then
        echo "❌ Error downloading $filename. Exiting script."
        exit 1
    fi
done


