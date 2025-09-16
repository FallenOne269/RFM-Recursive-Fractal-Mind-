// RFAI Peer Review Platform JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Initialize tab navigation
    initializeTabNavigation();
    
    // Initialize research domain expansion
    initializeDomainExpansion();
    
    // Initialize paper section navigation
    initializePaperSections();
    
    // Initialize review form
    initializeReviewForm();
    
    // Initialize smooth scrolling
    initializeSmoothScrolling();
    
    // Add loading animations
    setTimeout(addLoadingAnimations, 100);
    
    // Enhance accessibility
    enhanceAccessibility();
}

// Tab Navigation System - Fixed
function initializeTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    console.log('Initializing tab navigation with', tabButtons.length, 'buttons and', tabContents.length, 'content sections');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetTab = this.getAttribute('data-tab');
            console.log('Tab clicked:', targetTab);
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Find and activate the corresponding content
            const targetContent = document.getElementById(targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
                console.log('Activated tab content:', targetTab);
                
                // Trigger animations for the active tab
                setTimeout(() => animateTabContent(targetContent), 50);
            } else {
                console.error('Target content not found for tab:', targetTab);
            }
        });
    });
    
    // Ensure initial state is correct
    const activeButton = document.querySelector('.tab-button.active');
    if (activeButton) {
        const initialTab = activeButton.getAttribute('data-tab');
        const initialContent = document.getElementById(initialTab);
        if (initialContent && !initialContent.classList.contains('active')) {
            initialContent.classList.add('active');
        }
    }
}

// Research Domain Expansion
function initializeDomainExpansion() {
    const domainCards = document.querySelectorAll('.domain-card');
    
    domainCards.forEach(card => {
        const expandBtn = card.querySelector('.expand-btn');
        const details = card.querySelector('.domain-details');
        
        if (expandBtn && details) {
            expandBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                const isExpanded = !details.classList.contains('hidden');
                
                if (isExpanded) {
                    // Collapse
                    details.classList.add('hidden');
                    expandBtn.textContent = '+';
                    expandBtn.setAttribute('aria-expanded', 'false');
                } else {
                    // Expand
                    details.classList.remove('hidden');
                    expandBtn.textContent = '−';
                    expandBtn.setAttribute('aria-expanded', 'true');
                    
                    // Animate the expansion
                    animateDomainExpansion(details);
                }
            });
        }
    });
}

// Paper Section Navigation - Fixed
function initializePaperSections() {
    const sectionButtons = document.querySelectorAll('.section-btn');
    const paperSections = document.querySelectorAll('.paper-section');
    
    console.log('Initializing paper sections with', sectionButtons.length, 'buttons and', paperSections.length, 'sections');
    
    sectionButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetSection = this.getAttribute('data-section');
            console.log('Section clicked:', targetSection);
            
            // Remove active class from all buttons and sections
            sectionButtons.forEach(btn => btn.classList.remove('active'));
            paperSections.forEach(section => section.classList.remove('active'));
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Find and activate the corresponding section
            const targetContent = document.getElementById(targetSection);
            if (targetContent) {
                targetContent.classList.add('active');
                console.log('Activated paper section:', targetSection);
                
                // Animate section transition
                setTimeout(() => animateSectionTransition(targetContent), 50);
            } else {
                console.error('Target section not found:', targetSection);
            }
        });
    });
}

// Review Form Handling
function initializeReviewForm() {
    const reviewForm = document.querySelector('.review-form');
    
    if (reviewForm) {
        reviewForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Collect form data
            const formData = collectReviewData();
            
            // Validate form
            if (validateReviewForm(formData)) {
                // Simulate form submission
                submitReview(formData);
            } else {
                showValidationErrors();
            }
        });
        
        // Add real-time validation for rating scales
        const radioInputs = reviewForm.querySelectorAll('input[type="radio"]');
        radioInputs.forEach(input => {
            input.addEventListener('change', function() {
                updateReviewProgress();
            });
        });
        
        // Initialize review progress
        updateReviewProgress();
    }
}

// Smooth Scrolling
function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href.length > 1) {
                e.preventDefault();
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

// Animation Functions
function addLoadingAnimations() {
    // Add fade-in animation to dashboard cards
    const dashboardCards = document.querySelectorAll('.dashboard-card');
    dashboardCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 150);
    });
}

function animateTabContent(content) {
    // Add entrance animation for tab content
    const elements = content.querySelectorAll('.dashboard-card, .domain-card, .chart-section, .paper-sections, .review-interface, .references-container');
    
    elements.forEach((element, index) => {
        if (element) {
            element.style.opacity = '0';
            element.style.transform = 'translateY(15px)';
            
            setTimeout(() => {
                element.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }, index * 100);
        }
    });
}

function animateDomainExpansion(details) {
    details.style.maxHeight = '0';
    details.style.opacity = '0';
    details.style.overflow = 'hidden';
    details.style.transition = 'max-height 0.4s ease, opacity 0.3s ease';
    
    setTimeout(() => {
        details.style.maxHeight = '500px';
        details.style.opacity = '1';
    }, 10);
    
    setTimeout(() => {
        details.style.maxHeight = 'none';
        details.style.overflow = 'visible';
    }, 450);
}

function animateSectionTransition(section) {
    section.style.opacity = '0';
    section.style.transform = 'translateX(20px)';
    
    setTimeout(() => {
        section.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        section.style.opacity = '1';
        section.style.transform = 'translateX(0)';
    }, 10);
}

// Review Form Functions
function collectReviewData() {
    const form = document.querySelector('.review-form');
    const data = {
        ratings: {},
        comments: {},
        recommendation: ''
    };
    
    if (!form) return data;
    
    // Collect ratings
    const ratingInputs = form.querySelectorAll('input[type="radio"]:checked');
    ratingInputs.forEach(input => {
        data.ratings[input.name] = input.value;
    });
    
    // Collect text areas
    const textAreas = form.querySelectorAll('textarea');
    textAreas.forEach(textarea => {
        const label = textarea.previousElementSibling ? textarea.previousElementSibling.textContent : 'comment';
        data.comments[label] = textarea.value.trim();
    });
    
    // Collect recommendation
    const recommendationSelect = form.querySelector('select');
    if (recommendationSelect) {
        data.recommendation = recommendationSelect.value;
    }
    
    return data;
}

function validateReviewForm(data) {
    let isValid = true;
    const errors = [];
    
    // Check if all ratings are provided
    const requiredRatings = ['technical', 'novelty', 'theory', 'validation'];
    requiredRatings.forEach(rating => {
        if (!data.ratings[rating]) {
            errors.push(`Please provide a rating for ${rating.charAt(0).toUpperCase() + rating.slice(1)}`);
            isValid = false;
        }
    });
    
    // Check if recommendation is selected
    if (!data.recommendation) {
        errors.push('Please select a publication recommendation');
        isValid = false;
    }
    
    // Check if at least one comment field is filled
    const hasComments = Object.values(data.comments).some(comment => comment.length > 0);
    if (!hasComments) {
        errors.push('Please provide at least one detailed comment');
        isValid = false;
    }
    
    if (!isValid) {
        console.log('Validation errors:', errors);
    }
    
    return isValid;
}

function submitReview(data) {
    // Show loading state
    const submitBtn = document.querySelector('.review-form button[type="submit"]');
    if (!submitBtn) return;
    
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Submitting...';
    submitBtn.disabled = true;
    
    // Simulate API call
    setTimeout(() => {
        // Show success message
        showSubmissionSuccess();
        
        // Reset button
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
        
        // Reset form
        const form = document.querySelector('.review-form');
        if (form) {
            form.reset();
            updateReviewProgress();
        }
        
        console.log('Review submitted:', data);
    }, 2000);
}

function showSubmissionSuccess() {
    // Remove existing messages
    const existingMessage = document.querySelector('.submission-success');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Create success message
    const successMessage = document.createElement('div');
    successMessage.className = 'submission-success';
    successMessage.innerHTML = `
        <div style="
            background: rgba(var(--color-success-rgb), 0.1);
            border: 1px solid rgba(var(--color-success-rgb), 0.3);
            color: var(--color-success);
            padding: var(--space-16);
            border-radius: var(--radius-base);
            margin-bottom: var(--space-20);
            text-align: center;
            font-weight: var(--font-weight-medium);
        ">
            ✓ Review submitted successfully! Thank you for your valuable feedback.
        </div>
    `;
    
    const reviewInterface = document.querySelector('.review-interface');
    if (reviewInterface) {
        reviewInterface.insertBefore(successMessage, reviewInterface.firstChild);
        
        // Remove success message after 5 seconds
        setTimeout(() => {
            if (successMessage.parentNode) {
                successMessage.parentNode.removeChild(successMessage);
            }
        }, 5000);
    }
}

function showValidationErrors() {
    // Remove existing error messages
    const existingError = document.querySelector('.validation-errors');
    if (existingError) {
        existingError.remove();
    }
    
    // Create error message
    const errorMessage = document.createElement('div');
    errorMessage.className = 'validation-errors';
    errorMessage.innerHTML = `
        <div style="
            background: rgba(var(--color-error-rgb), 0.1);
            border: 1px solid rgba(var(--color-error-rgb), 0.3);
            color: var(--color-error);
            padding: var(--space-16);
            border-radius: var(--radius-base);
            margin-bottom: var(--space-20);
            text-align: center;
            font-weight: var(--font-weight-medium);
        ">
            ⚠ Please complete all required fields before submitting your review.
        </div>
    `;
    
    const reviewInterface = document.querySelector('.review-interface');
    if (reviewInterface) {
        reviewInterface.insertBefore(errorMessage, reviewInterface.firstChild);
        
        // Remove error message after 5 seconds
        setTimeout(() => {
            if (errorMessage.parentNode) {
                errorMessage.parentNode.removeChild(errorMessage);
            }
        }, 5000);
    }
}

function updateReviewProgress() {
    const form = document.querySelector('.review-form');
    if (!form) return;
    
    const totalCriteria = 4; // Number of rating criteria
    const completedRatings = form.querySelectorAll('input[type="radio"]:checked').length;
    const progress = (completedRatings / totalCriteria) * 100;
    
    // Update submit button state based on progress
    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
        if (progress === 100) {
            submitBtn.style.opacity = '1';
            submitBtn.style.cursor = 'pointer';
        } else {
            submitBtn.style.opacity = '0.7';
            submitBtn.style.cursor = 'default';
        }
    }
}

// Accessibility Enhancements
function enhanceAccessibility() {
    // Add ARIA labels and roles for tabs
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach((button, index) => {
        button.setAttribute('role', 'tab');
        const tabId = button.getAttribute('data-tab');
        button.setAttribute('aria-controls', tabId);
        button.setAttribute('id', tabId + '-tab');
        button.setAttribute('tabindex', button.classList.contains('active') ? '0' : '-1');
    });
    
    tabContents.forEach(content => {
        content.setAttribute('role', 'tabpanel');
        content.setAttribute('aria-labelledby', content.id + '-tab');
    });
    
    // Add expand/collapse accessibility
    const expandButtons = document.querySelectorAll('.expand-btn');
    expandButtons.forEach((button, index) => {
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('aria-label', 'Expand domain details');
        button.setAttribute('id', 'expand-btn-' + index);
    });
}

// Keyboard Navigation Support
document.addEventListener('keydown', function(e) {
    // Add keyboard navigation for tabs
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        const activeTab = document.querySelector('.tab-button.active');
        if (activeTab && document.activeElement === activeTab) {
            const allTabs = Array.from(document.querySelectorAll('.tab-button'));
            const currentIndex = allTabs.indexOf(activeTab);
            let nextIndex;
            
            if (e.key === 'ArrowRight') {
                nextIndex = (currentIndex + 1) % allTabs.length;
            } else {
                nextIndex = (currentIndex - 1 + allTabs.length) % allTabs.length;
            }
            
            e.preventDefault();
            allTabs[nextIndex].click();
            allTabs[nextIndex].focus();
        }
    }
});

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Performance Optimization
function optimizePerformance() {
    // Lazy load images that are not initially visible
    if ('IntersectionObserver' in window) {
        const images = document.querySelectorAll('img[data-src]');
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    observer.unobserve(img);
                }
            });
        });
        
        images.forEach(img => imageObserver.observe(img));
    }
}

// Initialize performance optimizations
document.addEventListener('DOMContentLoaded', optimizePerformance);

// Export functions for potential external use
window.RFAIApp = {
    initializeApp,
    collectReviewData,
    validateReviewForm,
    animateTabContent,
    showSubmissionSuccess
};