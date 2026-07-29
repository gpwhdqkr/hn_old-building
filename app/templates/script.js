document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const uploadText = document.getElementById('uploadText');
    const btnDiagnose = document.getElementById('btnDiagnose');
    const systemStatus = document.getElementById('systemStatus');
    const previewImg = document.getElementById('previewImg');
    const gridBg = document.getElementById('gridBg');
    const logZone = document.getElementById('logZone');
    const uploadZone = document.getElementById('uploadZone');
    const laserLine = document.getElementById('laserLine');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultOverlay = document.getElementById('resultOverlay');
    const resultPanel = document.getElementById('resultPanel');

    let isFinished = false;

    // 드래그 앤 드롭 브라우저 기본 동작 방지
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // 드래그 시 시각 피드백
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => uploadBox.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadBox.addEventListener(eventName, () => uploadBox.classList.remove('dragover'), false);
    });

    // 파일 드롭 핸들러
    uploadBox.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 업로드 상자 클릭 시 파일 탐색기 연동
    uploadBox.addEventListener('click', () => fileInput.click());

    // [핵심 수정] 파일 탐색기 창이 떠 있는 상태에서도 선택 즉시 반응하도록 이벤트 최적화
    fileInput.addEventListener('input', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // 파일 로드 및 UI 상태 즉시 갱신 함수 (비동기 처리로 탐색기 정체 현상 해결)
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 업로드 가능합니다!');
            return;
        }

        const reader = new FileReader();
        
        // 브라우저 백그라운드 렌더링 스레드 활용
        reader.onload = (event) => {
            requestAnimationFrame(() => {
                previewImg.src = event.target.result;
                previewImg.classList.remove('hide');
                gridBg.classList.add('hide');

                systemStatus.textContent = '[READY TO SCAN]';
                systemStatus.style.color = '#10b981';
                
                uploadText.innerHTML = `<strong>${file.name}</strong><br>스캔 준비 완료`;
                
                // 탐색기 창 차단 현상을 우회하여 즉각 활성화
                btnDiagnose.removeAttribute('disabled');
                btnDiagnose.classList.add('active');
                btnDiagnose.textContent = '진단 시작';
                isFinished = false;
            });
        };
        reader.readAsDataURL(file);
    }

    // 버튼 동작 컨트롤러
    btnDiagnose.addEventListener('click', () => {
        if (isFinished) {
            resetSystem();
            return;
        }

        btnDiagnose.setAttribute('disabled', 'true');
        btnDiagnose.classList.remove('active');
        uploadZone.classList.add('hide');
        logZone.classList.remove('hide');
        laserLine.classList.remove('hide');
        
        systemStatus.textContent = '[ANALYZING DATA...]';
        systemStatus.style.color = '#38bdf8';

        let progress = 0;
        const interval = setInterval(() => {
            progress += 2;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `[ANALYZING... ${progress}%]`;

            if (progress >= 100) {
                clearInterval(interval);
                showResults();
            }
        }, 50);
    });

    // 결과 출력
    function showResults() {
        laserLine.classList.add('hide'); 
        systemStatus.textContent = '[DIAGNOSIS COMPLETE]';
        systemStatus.style.color = '#ef4444';

        resultOverlay.classList.remove('hide');
        resultPanel.classList.remove('hide');

        isFinished = true;
        btnDiagnose.removeAttribute('disabled');
        btnDiagnose.classList.add('active');
        btnDiagnose.textContent = '다시 진단하기';
    }

    // 초기화 리셋
    function resetSystem() {
        isFinished = false;
        fileInput.value = ''; 
        previewImg.src = '';
        previewImg.classList.add('hide');
        gridBg.classList.remove('hide');

        progressBar.style.width = '0%';
        progressText.textContent = '[ANALYZING... 0%]';

        resultOverlay.classList.add('hide');
        resultPanel.classList.add('hide');
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