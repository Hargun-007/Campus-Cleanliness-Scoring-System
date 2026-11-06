// Campus Cleanliness Scoring System - Client JavaScript
document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const previewInfo = document.getElementById("preview-info");
  const removeBtn = document.getElementById("remove-preview-btn");
  const uploadForm = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");
  const loadingOverlay = document.getElementById("loading-overlay");
  const confSlider = document.getElementById("conf-slider");
  const confValue = document.getElementById("conf-value");
  const dropzoneContent = document.getElementById("dropzone-content");

  // Confidence slider live update
  if (confSlider && confValue) {
    confSlider.addEventListener("input", (e) => {
      confValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Handle Drag & Drop
  if (dropzone && fileInput) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        fileInput.files = files;
        handleFileSelect(files[0]);
      }
    });

    // Clicking dropzone opens file dialog
    dropzone.addEventListener("click", (e) => {
      if (e.target !== removeBtn && !removeBtn?.contains(e.target)) {
        fileInput.click();
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file) return;

    // Check size (16 MB)
    if (file.size > 16 * 1024 * 1024) {
      alert("Selected file exceeds the 16 MB maximum size limit.");
      clearFileSelection();
      return;
    }

    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      alert("Unsupported file format. Please upload a JPG, PNG, or WEBP image.");
      clearFileSelection();
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImage) {
        previewImage.src = e.target.result;
      }
      if (previewInfo) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        previewInfo.textContent = `${file.name} (${sizeMb} MB)`;
      }
      if (previewContainer) {
        previewContainer.style.display = "block";
      }
      if (dropzoneContent) {
        dropzoneContent.style.display = "none";
      }
      if (submitBtn) {
        submitBtn.disabled = false;
      }
    };
    reader.readAsDataURL(file);
  }

  function clearFileSelection() {
    if (fileInput) fileInput.value = "";
    if (previewContainer) previewContainer.style.display = "none";
    if (dropzoneContent) dropzoneContent.style.display = "block";
    if (submitBtn) submitBtn.disabled = true;
  }

  if (removeBtn) {
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearFileSelection();
    });
  }

  // Handle Form Submission and Loading Overlay
  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      if (loadingOverlay) {
        loadingOverlay.style.display = "flex";
      }
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = "<span>Analyzing Image...</span>";
      }
    });
  }

  // Handle Sample Image Clicks
  const sampleCards = document.querySelectorAll(".sample-card");
  sampleCards.forEach((card) => {
    card.addEventListener("click", () => {
      const sampleName = card.getAttribute("data-sample-name");
      if (!sampleName) return;

      const sampleInput = document.getElementById("sample-name-input");
      if (sampleInput && uploadForm) {
        sampleInput.value = sampleName;
        if (loadingOverlay) {
          loadingOverlay.style.display = "flex";
        }
        uploadForm.submit();
      }
    });
  });
});

