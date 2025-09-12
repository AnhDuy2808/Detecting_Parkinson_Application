// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Lấy các đối tượng HTML cần thiết
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const audioPlayback = document.getElementById('audioPlayback');
    const audioFile = document.getElementById('audioFile');
    const diagnoseButton = document.getElementById('diagnoseButton');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results');
    
    // Cập nhật để lấy các phần tử mới
    const healthyPercentageSpan = document.getElementById('healthy-percentage');
    const parkinsonPercentageSpan = document.getElementById('parkinson-percentage');
    const healthyProgressBar = document.querySelector('.healthy-bar');
    const parkinsonProgressBar = document.querySelector('.parkinson-bar');

    let mediaRecorder;
    let audioChunks = [];
    let audioBlob = null;

    // --- LOGIC GHI ÂM ---
    recordButton.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.start();
            audioChunks = [];
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });

            recordButton.disabled = true;
            stopButton.disabled = false;
            diagnoseButton.disabled = true;
            audioFile.value = null;
            audioBlob = null;

            // Ẩn kết quả cũ khi bắt đầu ghi âm/chẩn đoán mới
            resultsContainer.style.display = 'none';

        } catch (err) {
            alert('Lỗi khi truy cập micro. Vui lòng kiểm tra và cấp quyền truy cập micro cho trình duyệt.');
            console.error("Lỗi micro:", err);
        }
    });

    stopButton.addEventListener('click', () => {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop()); 
        
        mediaRecorder.addEventListener('stop', () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            
            stopButton.disabled = true;
            recordButton.disabled = false;
            diagnoseButton.disabled = false;
        });
    });

    // --- LOGIC UPLOAD FILE ---
    audioFile.addEventListener('change', () => {
        if (audioFile.files.length > 0) {
            diagnoseButton.disabled = false;
            audioBlob = null;
            audioPlayback.src = URL.createObjectURL(audioFile.files[0]);
            audioPlayback.load();
            // Ẩn kết quả cũ khi chọn file mới
            resultsContainer.style.display = 'none';
        }
    });

    // --- LOGIC CHẨN ĐOÁN ---
    diagnoseButton.addEventListener('click', async () => {
        let audioData = null;
        let fileName = "recording.wav";

        if (audioBlob) {
            audioData = audioBlob;
        } else if (audioFile.files.length > 0) {
            audioData = audioFile.files[0];
            fileName = audioFile.files[0].name;
        } else {
            alert("Vui lòng ghi âm hoặc chọn một tệp âm thanh.");
            return;
        }

        const formData = new FormData();
        formData.append('audio_data', audioData, fileName);

        loader.style.display = 'block';
        resultsContainer.style.display = 'none'; // Đảm bảo ẩn kết quả cũ
        diagnoseButton.disabled = true;

        try {
            const response = await fetch('/predict', { 
                method: 'POST', 
                body: formData 
            });

            const result = await response.json();

            if (!response.ok) { 
                throw new Error(result.error || 'Lỗi không xác định từ máy chủ'); 
            }
            displayResults(result);

        } catch (error) {
            console.error('Lỗi khi chẩn đoán:', error);
            alert(`Đã xảy ra lỗi: ${error.message}`);
        } finally {
            loader.style.display = 'none';
            diagnoseButton.disabled = false;
        }
    });

    function displayResults(result) {
        resultsContainer.style.display = 'block';
        
        const healthy_perc = parseFloat(result.healthy_percentage.replace('%', ''));
        const parkinson_perc = parseFloat(result.parkinson_percentage.replace('%', ''));

        healthyPercentageSpan.textContent = result.healthy_percentage;
        parkinsonPercentageSpan.textContent = result.parkinson_percentage;

        // Cập nhật chiều rộng của thanh tiến trình
        healthyProgressBar.style.width = `${healthy_perc}%`;
        parkinsonProgressBar.style.width = `${parkinson_perc}%`;

        // Thêm animation cho thanh tiến trình
        healthyProgressBar.style.transition = 'width 1s ease-in-out';
        parkinsonProgressBar.style.transition = 'width 1s ease-in-out';
    }
});