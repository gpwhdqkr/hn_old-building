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
            handleFileValidation(e.dataTransfer.files[0]); // 첫 번째 파일만 안전하게 전달
        }
    });

    uploadBox.addEventListener('click', () => fileInput.click());
    
    // 🔒 [중복 바인딩 버그 교정] input과 change의 이중 호출을 막기 위해 가장 확실한 change 하나로 통합 가동
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileValidation(e.target.files[0]);
        }
    });

    function handleFileValidation(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 업로드 가능합니다!');
            resetSystem();
            return;
        }

        const fileName = file.name.toLowerCase();
        const hasAllowedExtension = fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png');

        if (!hasAllowedExtension) {
            alert('허용되지 않은 파일 형식입니다. JPG, JPEG, PNG 이미지만 업로드해 주세요.');
            resetSystem();
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const w = img.width;
                const h = img.height;

                if (w > 1024 || h > 1024) {
                    alert(`이미지 해상도가 너무 큽니다. 가로 및 세로가 1024픽셀 이하인 사진을 올려주세요. (업로드된 크기: ${w}x${h})`);
                    resetSystem();
                    return;
                }

                selectedFileBlob = file;
                renderPreview(e.target.result, file.name);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function renderPreview(imageSrc, fileName) {
        requestAnimationFrame(() => {
            previewImg.src = imageSrc;
            previewImg.classList.remove('hide');
            resultImg.classList.add('hide');
            gridBg.classList.add('hide');
            systemStatus.textContent = '[READY TO SCAN]';
            systemStatus.style.color = '#10b981';
            uploadText.innerHTML = `<strong>${fileName}</strong><br>스캔 준비 완료`;
            btnDiagnose.removeAttribute('disabled');
            btnDiagnose.classList.add('active');
            btnDiagnose.textContent = '진단 시작';
            isFinished = false;
            
            logZone.querySelector('.code-log').innerHTML = `
                <p>> [INFO] Initializing ConvNeXt-Tiny Engine...</p>
                <p>> [INFO] Loading weights into CUDA/CPU context...</p>
                <p>> [DATA] Transferring payload to model tensor...</p>
                <p class="blink">> [COMPUTING] ConvNeXt-Tiny inference running...</p>
            `;
        });
    }

    btnDiagnose.addEventListener('click', () => {
        if (isFinished) { resetSystem(); return; }
        if (!selectedFileBlob) { resetSystem(); return; }

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
            progressText.textContent = `[ANALYZING... ${progress-4}%]`;

            if (progress >= 100 && isResponseReady) {
                clearInterval(scanInterval);
                injectBackendResult(responseHtmlText);
            }
        }, 50);

        fetch('/predict', { method: 'POST', body: payload })
        .then(res => res.text())
        .then(htmlResult => {
            if (htmlResult.includes('alert(')) {
                const match = htmlResult.match(/alert\("([^"]+)"\)/);
                const errorMsg = match ? match[1] : "서버 방어가드 조건에 위배되었습니다.";
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

        logZone.querySelector('.code-log').innerHTML = `
            <p>> [INFO] Initializing ConvNeXt-Tiny Engine...</p>
            <p>> [INFO] Loading weights into CUDA/CPU context...</p>
            <p>> [DATA] Transferring payload to model tensor...</p>
            <p>> [COMPUTING] LayerCAM tracking in progress...</p>
            <p style="color: #10b981;">> [COMPLETE] Architecture diagnostic logic executed.</p>
        `;

        const metaPipe = document.getElementById('backendUrls');
        if (!metaPipe) {
            alert("서버 결과 템플릿 파싱에 실패했습니다.");
            resetSystem();
            return;
        }

        const finalResultImgUrl = metaPipe.getAttribute('data-result');
        const finalStatus = metaPipe.getAttribute('data-status');

        resultImg.src = finalResultImgUrl;
        resultImg.classList.remove('hide');

        if (finalStatus && finalStatus.includes("불량")) {
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
        uploadText.innerHTML = `사진을 여기로 드래그하거나<br>클릭하여 업로드하세요<br>(50MB 이하, .png .jpg .jpeg 만 가능)<br>진단하고 싶은 하자가 정중앙에 위치한 사진 권장.`;
    }
});
