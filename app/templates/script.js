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
    const imageViewport = document.getElementById('imageViewport');
    const compareWrap = document.getElementById('compareWrap');
    const beforeImg = document.getElementById('beforeImg');
    const afterTag = document.getElementById('afterTag');
    const layerTabs = document.getElementById('layerTabs');
    const tabHeatmap = document.getElementById('tabHeatmap');
    const tabBox = document.getElementById('tabBox');

    let isFinished = false;
    let selectedFileBlob = null;

    // after 레이어로 쓸 이미지 주소 보관 (탭 전환 시 즉시 교체)
    const layerUrls = { heatmap: '', box: '' };
    const DEFAULT_SPLIT = 50;   // 분할선 초기 위치 (%)

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

    // =========================================================================
    // before/after 비교 슬라이더 제어
    // =========================================================================

    /** 분할선 위치(%) 적용 — CSS 변수 하나로 clip-path와 divider가 함께 움직인다. */
    function setSplit(percent) {
        const clamped = Math.max(0, Math.min(100, percent));
        compareWrap.style.setProperty('--split', `${clamped}%`);
    }

    /** 뷰포트(style.css의 .image-viewport 높이) 안에서 원본 비율을 유지하는 비교 상자
     *  크기를 계산해 넣는다. 두 레이어가 같은 상자를 100%로 채우므로 분할선이
     *  이미지 밖으로 새지 않는다. */
    function fitCompareWrap() {
        if (!beforeImg.naturalWidth || !beforeImg.naturalHeight) return;
        const boxWidth = imageViewport.clientWidth;
        const boxHeight = imageViewport.clientHeight;
        const aspect = beforeImg.naturalWidth / beforeImg.naturalHeight;

        let width = boxWidth;
        let height = boxWidth / aspect;
        if (height > boxHeight) {          // 세로 사진: 높이 기준으로 다시 맞춤
            height = boxHeight;
            width = boxHeight * aspect;
        }
        compareWrap.style.width = `${Math.round(width)}px`;
        compareWrap.style.height = `${Math.round(height)}px`;
    }

    /** 포인터 x좌표 → 분할선 % 변환 (마우스·터치·펜 공통) */
    function moveSplitToPointer(event) {
        const rect = compareWrap.getBoundingClientRect();
        if (!rect.width) return;
        setSplit(((event.clientX - rect.left) / rect.width) * 100);
    }

    // 드래그 상태는 자체 플래그로 관리한다 — setPointerCapture가 실패하는 환경에서도
    // 슬라이더가 멈추지 않도록 (캡처는 상자 밖으로 나갔을 때를 위한 보조 수단)
    let isDragging = false;

    compareWrap.addEventListener('pointerdown', (e) => {
        isDragging = true;
        try { compareWrap.setPointerCapture(e.pointerId); } catch (_) { /* 캡처 미지원 무시 */ }
        moveSplitToPointer(e);
    });
    compareWrap.addEventListener('pointermove', (e) => {
        if (isDragging) moveSplitToPointer(e);
    });
    ['pointerup', 'pointercancel', 'pointerleave'].forEach(name => {
        compareWrap.addEventListener(name, (e) => {
            isDragging = false;
            try { compareWrap.releasePointerCapture(e.pointerId); } catch (_) { /* 이미 해제됨 */ }
        });
    });
    // 브라우저 창 크기가 바뀌면 비교 상자도 다시 맞춘다
    window.addEventListener('resize', fitCompareWrap);

    /** 히트맵/박스 탭 전환 — after 레이어 이미지만 바꾸고 분할선은 유지한다. */
    function selectLayer(layer) {
        if (!layerUrls[layer]) return;               // 없는 레이어(예: 우수의 히트맵)는 무시
        resultImg.src = layerUrls[layer];
        tabHeatmap.classList.toggle('active', layer === 'heatmap');
        tabBox.classList.toggle('active', layer === 'box');
        afterTag.textContent = layer === 'heatmap' ? 'AFTER · 히트맵' : 'AFTER · 결함 박스';
    }

    [tabHeatmap, tabBox].forEach(tab => {
        tab.addEventListener('click', () => selectLayer(tab.dataset.layer));
    });

    function handleFileValidation(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 업로드 가능합니다!');
            resetSystem();
            return;
        }

        const fileName = file.name.toLowerCase();
        // 끝자리가 .jpg, .jpeg, .png, .jfif 중 하나인지 완벽히 체크
        const hasAllowedExtension = ['.jpg', '.jpeg', '.png', '.jfif'].some(ext => fileName.endsWith(ext));

        if (!hasAllowedExtension) {
            alert('허용되지 않은 파일 형식입니다. JPG, JPEG, PNG, JFIF 이미지만 업로드해 주세요.');
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
            // [추가] 이전 진단의 before/after 슬라이더와 레이어 탭도 함께 걷어낸다
            compareWrap.classList.add('hide');
            layerTabs.classList.add('hide');
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

        const finalOriginImgUrl = metaPipe.getAttribute('data-origin');
        const finalResultImgUrl = metaPipe.getAttribute('data-result');
        const finalHeatmapImgUrl = metaPipe.getAttribute('data-heatmap');
        const finalStatus = metaPipe.getAttribute('data-status');

        resultImg.src = finalResultImgUrl;
        resultImg.classList.remove('hide');

        // [추가] before/after 슬라이더용 레이어 주소 보관 (탭 전환 시 src만 교체)
        layerUrls.box = finalResultImgUrl;
        layerUrls.heatmap = finalHeatmapImgUrl || '';

        // 🔒 히트맵이 없는 경우(우수 판정/히트맵 생성 실패)는 탭을 감추고 박스 한 장만 보여준다
        const hasHeatmap = Boolean(layerUrls.heatmap);
        tabHeatmap.classList.toggle('hide', !hasHeatmap);
        tabBox.textContent = finalStatus.includes('불량') ? '결함 박스' : '판정 스탬프';
        layerTabs.classList.remove('hide');

        // before = 원본, after = 선택 레이어. 원본 크기를 알아야 비교 상자 비율이 잡히므로
        // 로드 완료(캐시 히트 포함) 시점에 fitCompareWrap을 한 번 호출한다.
        beforeImg.src = finalOriginImgUrl;
        selectLayer(hasHeatmap ? 'heatmap' : 'box');
        setSplit(DEFAULT_SPLIT);
        compareWrap.classList.remove('hide');

        if (beforeImg.complete && beforeImg.naturalWidth) {
            fitCompareWrap();
        } else {
            beforeImg.addEventListener('load', fitCompareWrap, { once: true });
        }

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
        // [추가] before/after 슬라이더·탭도 초기 상태로 되돌린다
        beforeImg.src = '';
        layerUrls.heatmap = '';
        layerUrls.box = '';
        setSplit(DEFAULT_SPLIT);
        compareWrap.classList.add('hide');
        layerTabs.classList.add('hide');
        tabHeatmap.classList.remove('hide');
        tabHeatmap.classList.add('active');
        tabBox.classList.remove('active');
        afterTag.textContent = 'AFTER · 히트맵';
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
        uploadText.innerHTML = `사진을 여기로 드래그하거나<br>클릭하여 업로드하세요<br><br>(50MB 이하, .png .jpg .jpeg .jfif 만 가능)<br>해상도 가로, 세로 1024 이하.<br>진단하고 싶은 하자가 정중앙에 위치한 사진 권장.`;
    }
});
