document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const uploadText = document.getElementById('uploadText');
    const btnDiagnose = document.getElementById('btnDiagnose');
    const systemStatus = document.getElementById('systemStatus');
    const previewImg = document.getElementById('previewImg');
    const resultImg = document.getElementById('resultImg');
    const gridBg = document.getElementById('gridBg');
    const logZone = document.getElementById('logZone');
    const uploadZone = document.getElementById('uploadZone');
    const laserLine = document.getElementById('laserLine');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const dynamicResult = document.getElementById('dynamicResult');

    let isFinished = false;
    let selectedFileBlob = null;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
        uploadBox.addEventListener(name, (e) => { e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter', 'dragover'].forEach(name => {
        uploadBox.addEventListener(name, () => uploadBox.classList.add('dragover'));
    });
    ['dragleave', 'drop'].forEach(name => {
        uploadBox.addEventListener(name, () => uploadBox.classList.remove('dragover'));
    });

    uploadBox.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) {
            selectedFileBlob = e.dataTransfer.files[0]; // 단일 바이너리 객체 추출 고정
            handlePreview(selectedFileBlob);
        }
    });

    uploadBox.addEventListener('click', () => fileInput.click());
    
    const onFileSelect = (e) => {
        if (e.target.files.length > 0) {
            selectedFileBlob = e.target.files[0];
            handlePreview(selectedFileBlob);
        }
    };
    fileInput.addEventListener('input', onFileSelect);
    fileInput.addEventListener('change', onFileSelect);

    function handlePreview(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 업로드 가능합니다!');
            return;
        }
        const reader = new FileReader();
        reader.onload = (event) => {
            requestAnimationFrame(() => {
                previewImg.src = event.target.result;
                previewImg.classList.remove('hide');
                resultImg.classList.add('hide');
                gridBg.classList.add('hide');
                systemStatus.textContent = '[READY TO SCAN]';
                systemStatus.style.color = '#10b981';
                uploadText.innerHTML = `<strong>${file.name}</strong><br>스캔 준비 완료`;
                btnDiagnose.removeAttribute('disabled');
                btnDiagnose.classList.add('active');
                btnDiagnose.textContent = '진단 시작';
                isFinished = false;
            });
        };
        reader.readAsDataURL(file);
    }

    btnDiagnose.addEventListener('click', () => {
        if (isFinished) { resetSystem(); return; }
        if (!selectedFileBlob) return;

        btnDiagnose.setAttribute('disabled', 'true');
        btnDiagnose.classList.remove('active');
        uploadZone.classList.add('hide');
        logZone.classList.remove('hide');
        laserLine.classList.remove('hide');
        systemStatus.textContent = '[RUNNING MODEL INFERENCE...]';
        systemStatus.style.color = '#38bdf8';

        const payload = new FormData();
        payload.append('house_image', selectedFileBlob);

        let progress = 0;
        let responseHtmlText = null;
        let isResponseReady = false;

        const scanInterval = setInterval(() => {
            progress += 4;
            if (progress > 96 && !isResponseReady) progress = 96; 
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `[ANALYZING... ${progress}%]`;

            if (progress >= 100 && isResponseReady) {
                clearInterval(scanInterval);
                injectBackendResult(responseHtmlText);
            }
        }, 50);

        fetch('/predict', { method: 'POST', body: payload })
        .then(res => res.text()) 
        .then(htmlResult => {
            if (htmlResult.includes('alert("')) {
                const parts = htmlResult.split('alert("');
                const errorMsg = parts[1].split('")')[0];
                throw new Error(errorMsg);
            }
            responseHtmlText = htmlResult;
            isResponseReady = true;
            progress = 100;
        })
        .catch(error => {
            clearInterval(scanInterval);
            alert(error.message);
            resetSystem();
        });
    });

    function injectBackendResult(htmlContent) {
        laserLine.classList.add('hide');
        previewImg.classList.add('hide');
        dynamicResult.innerHTML = htmlContent;
        dynamicResult.classList.remove('hide');

        const metaPipe = document.getElementById('backendUrls');
        const finalResultImgUrl = metaPipe.getAttribute('data-result');
        const finalStatus = metaPipe.getAttribute('data-status');

        // 🛠️ [경로 연동 무결성 처리] main.py가 슬래시를 이미 포함해서 반환하므로 중복 슬래시 방지 적용
        resultImg.src = finalResultImgUrl;
        resultImg.classList.remove('hide');

        if (finalStatus.includes("불량")) {
            systemStatus.textContent = '[DIAGNOSIS COMPLETE: DEFECT DETECTED]';
            systemStatus.style.color = '#ef4444';
        } else {
            systemStatus.textContent = '[DIAGNOSIS COMPLETE: SECURE]';
            systemStatus.style.color = '#10b981';
        }

        isFinished = true;
        btnDiagnose.removeAttribute('disabled');
        btnDiagnose.classList.add('active');
        btnDiagnose.textContent = '다시 진단하기';
    }

    function resetSystem() {
        isFinished = false;
        selectedFileBlob = null;
        fileInput.value = ''; 
        previewImg.src = '';
        resultImg.src = '';
        previewImg.classList.add('hide');
        resultImg.classList.add('hide');
        gridBg.classList.remove('hide');
        progressBar.style.width = '0%';
        progressText.textContent = '[ANALYZING... 0%]';
        dynamicResult.classList.add('hide');
        dynamicResult.innerHTML = '';
        logZone.classList.add('hide');
        uploadZone.classList.remove('hide');
        btnDiagnose.setAttribute('disabled', 'true');
        btnDiagnose.classList.remove('active');
        btnDiagnose.textContent = '진단 시작';
        systemStatus.textContent = '[SYSTEM READY: AWAITING INPUT]';
        systemStatus.style.color = '#00f2fe';
        uploadText.innerHTML = '사진을 여기로 드래그하거나<br>클릭하여 업로드하세요';
    }
});