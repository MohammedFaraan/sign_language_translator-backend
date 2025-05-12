import requests
import time
import sys
import os

def test_server_connection(base_url="http://localhost:5000"):
    """Test if the server is running and accessible"""
    print(f"Testing connection to server at {base_url}...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Server connection successful! Status code: {response.status_code}")
            return True
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_video_upload(video_path, base_url="http://localhost:5000"):
    """Test the video upload API endpoint"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    api_url = f"{base_url}/api/process-video"
    print(f"Testing video upload to {api_url}...")
    print(f"Using video file: {video_path} ({os.path.getsize(video_path)/1024/1024:.2f} MB)")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
            print("Uploading video... (this may take a while)")
            start_time = time.time()
            response = requests.post(api_url, files=files, timeout=180)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ Upload successful! Response received in {elapsed_time:.2f} seconds")
                try:
                    data = response.json()
                    print("\nDetected Signs:")
                    for sign, count in data.get('detected_signs', {}).items():
                        print(f"  - {sign}: {count} time(s)")
                    print(f"\nTotal signs: {data.get('total_signs', 0)}")
                    print(f"Unique signs: {data.get('unique_signs', 0)}")
                    return True
                except Exception as e:
                    print(f"❌ Error parsing response: {str(e)}")
                    print(f"Response content: {response.text[:500]}...")
                    return False
            else:
                print(f"❌ Upload failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        return False

if __name__ == "__main__":
    # Set default URL
    base_url = "http://localhost:5000"
    
    # Check server connection
    if not test_server_connection(base_url):
        print("\nServer connection failed. Make sure the Flask app is running:")
        print("  python app.py")
        sys.exit(1)
    
    # Test video upload if a file path is provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        test_video_upload(video_path, base_url)
    else:
        print("\nTo test video upload, provide a video file path:")
        print("  python test_connection.py path/to/video.mp4") 