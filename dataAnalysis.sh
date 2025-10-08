# written with chat-gpt
#!/bin/bash

EXEC="./contrast_enhancer"
IMG_DIR="../images/Foggy_Cityscapes/Dense_Fog"
MAX_ITER=70

cd "$(dirname "$0")/bin" || exit 1

for img in "$IMG_DIR"/*.png; do
    echo "Processing $img ..."
    $EXEC "$img" $MAX_ITER
done
