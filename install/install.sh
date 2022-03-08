#!/bin/bash

rustc -V &> /dev/null
if [ "$?" != "0" ]
then
    echo 'Please install Rust using rustup.'
    exit 1
else
    echo 'Rust found!'
fi

echo 'Checking for OpenCV...'
# Check for OpenCV 4.5 or above.
if opencv_version =~ '^4\.[5-9]|[1-9]\d\.*|[5-9]\.' &> /dev/null
then
    echo "Found OpenCV!"
else
    echo "Compatible OpenCV not found."
    case "$(uname)" in
        Darwin)
            read -p "Install OpenCV with homebrew? (Y/N)" confirmation
            if [[ "$confirmation" =~ ^y|Y$ ]]
            then
                brew install opencv
                # Restart script.
                exec "$ScriptLoc"
            else
                echo 'Not installing OpenCV.'
                exit 1
            fi
            ;;
        Linux)
            echo 'Please install a compatible version of OpenCV (4.5+).'
            echo "If you're on a Jetson Nano, look for a guide on compiling it from source."
            exit 1
    esac
fi

if [[ "$(uname)" == "Linux" ]]
then
    # Someone might run this outside of the working directory.
    cd $(dirname "${BASH_SOURCE[0]}")/..

    read -p "Install service? Only do this on the actual Jetson nano. (Y/N)" confirmation

    if [[ "$confirmation" =~ ^y|Y$ ]]
    then
        vision_template="$(<install/4904_vision.service)"
        printf "${vision_template//FOLDER_PATH/$(pwd)}" | sudo tee /etc/systemd/system/4904_vision.service >/dev/null
        sudo cp install/4904_webcam.rules /etc/udev/rules.d
        echo 'Installed using the current folder location. If you move the vision folder, rerun this script.'

        read -p "Enable installed service? (Y/N)" confirmation
        if [[ "$confirmation" =~ ^y|Y$ ]]
        then
            sudo systemctl enable 4904_vision.service
            sudo systemctl start 4904_vision
        fi
    else
        echo 'Service not installed.'
    fi
fi

echo "Done!"
