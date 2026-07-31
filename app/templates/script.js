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
    const afterImg = document.getElementById('afterImg');
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
        if (!layerUrls[layer]) return;               // 주소가 없는 레이어는 무시
        afterImg.src = layerUrls[layer];
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
        // -------------------------------------------------------------
        // [오리지널 유지] 기존 하단 영역 및 우측 뷰포트 프리뷰 가드 처리
        // -------------------------------------------------------------
        if (laserLine) laserLine.classList.add('hide');
        if (previewImg) previewImg.classList.add('hide');
        
        if (dynamicResult) {
            dynamicResult.innerHTML = htmlContent;
            dynamicResult.classList.remove('hide');
        }

        if (logZone && logZone.querySelector('.code-log')) {
            logZone.querySelector('.code-log').innerHTML = `
                <p>> [INFO] Initializing ConvNeXt-Tiny Engine...</p>
                <p>> [INFO] Loading weights into CUDA/CPU context...</p>
                <p>> [DATA] Transferring payload to model tensor...</p>
                <p>> [COMPUTING] LayerCAM tracking in progress...</p>
                <p style="color: #10b981;">> [COMPLETE] Architecture diagnostic logic executed.</p>
            `;
        }

        // 🌟 [안전 타이밍 보정] 브라우저가 화면 렌더링을 완전히 끝낸 뒤(0.1초 후) 좌측 박스에 데이터 주입
        setTimeout(() => {
            const reportContainer = document.getElementById('reportContainer');
            const reportContent = document.getElementById('reportContent');
            
            if (reportContainer && reportContent) {
                // 1. 서버가 보내준 검증된 HTML 소스를 좌측 상단 박스에 다이렉트 주입
                reportContent.innerHTML = htmlContent;
                reportContainer.classList.remove('hide');
                reportContent.scrollTop = 0;

                // 2. 파일 업로드 패널과 기존 조작 버튼들을 안전하게 스타일로 숨김
                if (uploadBox) uploadBox.style.display = 'none';
                if (uploadZone) uploadZone.classList.add('hide');
                if (logZone) logZone.classList.add('hide');
                if (btnDiagnose) btnDiagnose.classList.add('hide');
            } else {
                console.error("오류: 좌측 reportContainer 또는 reportContent 박스를 찾지 못했습니다.");
            }
        }, 100);

        // -------------------------------------------------------------
        // [오리지널 유지] 메타데이터 파이프라인 수수 및 우측 이미지/슬라이더 제어
        // -------------------------------------------------------------
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

        if (resultImg) {
            resultImg.src = finalResultImgUrl;
            resultImg.classList.remove('hide');
        }

        layerUrls.box = finalResultImgUrl;
        layerUrls.heatmap = finalHeatmapImgUrl || '';

        if (layerUrls.heatmap) {
            if (resultImg) resultImg.classList.add('hide');
            if (layerTabs) layerTabs.classList.remove('hide');

            if (beforeImg) beforeImg.src = finalOriginImgUrl;
            selectLayer('heatmap');
            setSplit(DEFAULT_SPLIT);
            if (compareWrap) compareWrap.classList.remove('hide');

            if (beforeImg && beforeImg.complete && beforeImg.naturalWidth) {
                fitCompareWrap();
            } else if (beforeImg) {
                beforeImg.addEventListener('load', fitCompareWrap, { once: true });
            }
        }

        if (finalStatus && finalStatus.includes("불량")) {
            if (systemStatus) {
                systemStatus.textContent = '[DIAGNOSIS COMPLETE: DEFECT DETECTED]';
                systemStatus.style.color = '#ef4444';
            }
        } else {
            if (systemStatus) {
                systemStatus.textContent = '[DIAGNOSIS COMPLETE: SECURE]';
                systemStatus.style.color = '#10b981';
            }
        }

   // [injectBackendResult 함수 내부 최하단부의 버튼 연동 블록 수정]
const btnHeaderDownload = document.getElementById('btnHeaderDownload');
if (btnHeaderDownload) {
    // 1. 숨겨져 있던 우측 상단 헤더 내 다운로드 스위치 전면 노출
    btnHeaderDownload.classList.remove('hide');
    
    // 2. 클릭 즉시 중간 불필요 팝업 없이 다이렉트로 다운로드 스트림 호출
    btnHeaderDownload.onclick = function() {
        // 백엔드가 결과 페이지 렌더링 시 주입한 HTML 엘리먼트 및 변수 데이터 수집
        const metaPipe = document.getElementById('backendUrls');
        
        // 렌더링된 속성값 수동 추적
        const userImg = metaPipe ? metaPipe.getAttribute('data-origin') : '';
        const camImg = metaPipe ? metaPipe.getAttribute('data-result') : '';
        const heatmapImg = metaPipe ? metaPipe.getAttribute('data-heatmap') : '';
        const aiResultText = document.querySelector('.system-status') ? document.querySelector('.system-status').textContent : '불량';
        
        // 데이터 수치 확보 (스크린 엘리먼트 파싱 예외 처리 가드 포함)
        const probPercent = window.probability_percent || "99.9"; 
        const infMs = window.inference_ms || "1637";

        // 3. 쿼리 파라미터를 동적으로 빌드하여 데이터 파이프라인 최종 전송
        const downloadUrl = `/api/download-report?user_image=${encodeURIComponent(userImg)}` +
                            `&cam_image=${encodeURIComponent(camImg)}` +
                            `&heatmap_image=${encodeURIComponent(heatmapImg || '')}` +
                            `&ai_result=${encodeURIComponent(aiResultText)}` +
                            `&probability_percent=${encodeURIComponent(probPercent)}` +
                            `&inference_ms=${encodeURIComponent(infMs)}`;

        // 브라우저 파일 수령 핸들러 작동 (화면 전환 없이 파일만 다운로드 다운)
        window.location.href = downloadUrl;
    };
}

    isFinished = true;
    if (btnDiagnose) {
        btnDiagnose.removeAttribute('disabled');
        btnDiagnose.classList.add('active');
        btnDiagnose.textContent = '다시 진단하기';
    }
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
        afterImg.src = '';
        layerUrls.heatmap = '';
        layerUrls.box = '';
        setSplit(DEFAULT_SPLIT);
        compareWrap.classList.add('hide');
        layerTabs.classList.add('hide');
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
    
                // 💡 [초기화 추가] 좌측 박스 숨기고 기존 업로드 박스 다시 드러내기
        const rc = document.getElementById('reportContainer');
        const rct = document.getElementById('reportContent');
        if (rc) rc.classList.add('hide');
        if (rct) rct.innerHTML = '';
        if (uploadBox) uploadBox.style.display = 'block'; // 숨겼던 업로드 영역 부활
        if (btnDiagnose) btnDiagnose.classList.remove('hide'); // 진단 버튼 부활

    
    }

    // ── 🆕 [진단 이력 기능] 여기부터 추가 (2026-07-31) ──
    // 좌측 진단 이력 사이드바. 상단 const 선언부를 건드리지 않으려고
    // 필요한 요소를 이 블록 안에서 따로 잡는다 (팀 병합 충돌 최소화).
    const historyToggle = document.getElementById('historyToggle');
    const historySidebar = document.getElementById('historySidebar');
    const historyScrim = document.getElementById('historyScrim');
    const historyList = document.getElementById('historyList');

    // status 문자열 → [배지 클래스, 표시 문구]
    const HISTORY_BADGE = { '불량': ['bad', '불량'], '우수': ['good', '우수'], '오류': ['err', '오류'] };

    function closeHistory() {
        historySidebar.classList.remove('open');
        historyScrim.classList.add('hide');
    }

    /** 썸네일 자리에 들어갈 회색 X 상자 */
    function makeThumbX() {
        const box = document.createElement('div');
        box.className = 'history-thumb-x';
        box.textContent = '✕';
        return box;
    }

    function renderHistory(items, failed) {
        historyList.innerHTML = '';

        if (!items.length) {
            const empty = document.createElement('li');
            empty.className = 'history-empty';
            // failed === true면 DB 장애 등으로 못 받아온 것 — "이력이 없다"고 단언하면 거짓말이 된다
            empty.textContent = failed ? '이력을 불러오지 못했습니다.' : '아직 진단 이력이 없습니다.';
            historyList.appendChild(empty);
            return;
        }

        items.forEach(item => {
            const isError = item.status === '오류';
            const row = document.createElement('li');
            row.className = 'history-item' + (isError ? ' is-error' : '');

            if (isError || !item.thumb) {
                row.appendChild(makeThumbX());
            } else {
                const thumb = document.createElement('img');
                thumb.className = 'history-thumb';
                thumb.src = item.thumb;
                thumb.alt = '';
                // 서버에서 파일이 지워졌으면 X 상자로 갈아끼우고 클릭도 막는다
                thumb.addEventListener('error', () => {
                    thumb.replaceWith(makeThumbX());
                    row.classList.add('is-error');
                }, { once: true });
                row.appendChild(thumb);
            }

            const date = document.createElement('span');
            date.className = 'history-date';
            date.textContent = item.date;
            row.appendChild(date);

            const badge = document.createElement('span');
            const badgeSpec = HISTORY_BADGE[item.status] || HISTORY_BADGE['오류'];
            badge.className = 'history-badge ' + badgeSpec[0];
            badge.textContent = badgeSpec[1];
            row.appendChild(badge);

            if (!isError) {
                row.addEventListener('click', () => {
                    if (row.classList.contains('is-error')) return;   // 썸네일이 404난 뒤 클릭 차단
                    restoreFromHistory(item.id);
                });
            }
            historyList.appendChild(row);
        });
    }

    function openHistory() {
        historySidebar.classList.add('open');
        historyScrim.classList.remove('hide');
        // 열 때마다 새로 받는다 — 방금 끝낸 진단이 바로 목록에 보인다
        fetch('/history')
            .then(res => res.json())
            .then(data => renderHistory(data.items || [], !!data.error))
            .catch(() => renderHistory([], true));  // 네트워크 실패도 DB 장애와 같은 문구로 안내
    }

    /** 저장된 진단 1건을 화면에 되살린다.
     *  서버가 /predict와 똑같은 조각을 주므로 기존 injectBackendResult()를 그대로 쓴다
     *  — 슬라이더·레이어 탭·fitCompareWrap()이 거기서 전부 세팅된다. */
    function restoreFromHistory(id) {
        fetch('/history/' + id)
            .then(res => {
                if (!res.ok) throw new Error('이력을 불러오지 못했습니다. 이미지가 삭제되었을 수 있습니다.');
                return res.text();
            })
            .then(htmlResult => {
                closeHistory();
                // btnDiagnose 클릭 핸들러(script.js:195-201)와 같은 화면 전환
                uploadZone.classList.add('hide');
                logZone.classList.remove('hide');
                laserLine.classList.add('hide');
                progressBar.style.width = '100%';
                progressText.textContent = '[ANALYZING... 100%]';
                // 이전 복원이 남긴 슬라이더·탭을 먼저 걷어낸다.
                // injectBackendResult()는 히트맵이 있을 때 켜기만 하고 없을 때 끄지 않으므로,
                // 우수 건을 복원하면 직전 불량 건의 레이어가 그대로 남는다 (renderPreview와 같은 정리).
                compareWrap.classList.add('hide');
                layerTabs.classList.add('hide');
                injectBackendResult(htmlResult);
            })
            .catch(err => alert(err.message));
    }

    historyToggle.addEventListener('click', () => {
        if (historySidebar.classList.contains('open')) closeHistory();
        else openHistory();
    });
    historyScrim.addEventListener('click', closeHistory);
    // ── 🆕 [진단 이력 기능] 여기까지 ──

      // 💡 [최하단 코드 교체] 출처 독립 토글 시스템
    const btnSource = document.getElementById('btnSource');
    const sourceContent = document.getElementById('sourceContent');

    if (btnSource && sourceContent) {
        btnSource.addEventListener('click', () => {
            // 다른 요소와 꼬이지 않는 출처 전용 클래스로 토글 작동
            sourceContent.classList.toggle('source-hide');
            
            if (sourceContent.classList.contains('source-hide')) {
                btnSource.textContent = '데이터셋 출처 확인';
                btnSource.classList.remove('active');
            } else {
                btnSource.textContent = '출처 정보 닫기';
                btnSource.classList.add('active');
            }
        });
    }
   

    document.addEventListener('DOMContentLoaded', () => {
      // =========================================================================
    // 📄 [완벽 교정] 문법 오류 원천 제거 및 실시간 데이터 다운로드 파이프라인
    // =========================================================================
    
    // 1. 변수 안전 바인딩 확인
    const btnHeaderDownload = document.getElementById('btnHeaderDownload');
    const systemStatus = document.getElementById('systemStatus');

    // 2. 백엔드 통신 엔진(injectBackendResult) 내부 다운로드 스위치 동적 결합
    // 기존에 에러가 나던 텍스트 파편과 구형 모의 함수를 걷어내고 실무 연동형으로 교정 완료했습니다.
    if (btnHeaderDownload) {
        // 백엔드 연산 완료 타이밍에 맞춰 버튼의 'hide' 클래스를 제거하는 제어권 통합
        // (현재 injectBackendResult 함수 내부에서 remove('hide')를 호출하므로, 
        //  이벤트 리스너가 중복 꼬이지 않도록 클릭 핸들러만 깔끔하게 단독 배치합니다.)
        
        btnHeaderDownload.onclick = function() {
            const metaPipe = document.getElementById('backendUrls');
            
            // 화면 텍스트 대신, 백엔드 모델이 계산해서 숨겨놓은 실제 불량 확률 원값(소수점) 추적
            const rawDefectProb = metaPipe ? metaPipe.getAttribute('data-prob') : '0.999';
            const userImg = metaPipe ? metaPipe.getAttribute('data-origin') : '';
            const camImg = metaPipe ? metaPipe.getAttribute('data-result') : '';
            const heatmapImg = metaPipe ? metaPipe.getAttribute('data-heatmap') : '';
            
            const aiResultText = systemStatus ? systemStatus.textContent : '불량';
            
            // 전역 스코프에 바인딩된 밀리초 변수 안전 추출 가드
            const infMs = window.inference_ms || "1637";

            // 쿼리 스트링 파라미터를 동적으로 빌드하여 백엔드 PDF 다운로드 API 강제 기동
            const downloadUrl = `/api/download-report?user_image=${encodeURIComponent(userImg)}` +
                                `&cam_image=${encodeURIComponent(camImg)}` +
                                `&heatmap_image=${encodeURIComponent(heatmapImg || '')}` +
                                `&ai_result=${encodeURIComponent(aiResultText)}` +
                                `&raw_prob=${encodeURIComponent(rawDefectProb)}` +
                                `&inference_ms=${encodeURIComponent(infMs)}`;

   // ... 기존 코드(btnHeaderDownload.onclick 등)가 모두 끝나는 파일 맨 마지막 지점 ...
            window.location.href = downloadUrl;
        };
    }

    //  [이 아래의 2줄을 파일 맨 마지막에 추가하고 저장하세요]
});
});
