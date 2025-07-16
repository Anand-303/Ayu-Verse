-- Create user_dosha table to store dosha test results
CREATE TABLE IF NOT EXISTS user_dosha (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    dosha_type ENUM('Vata', 'Pitta', 'Kapha') NOT NULL,
    confidence_scores JSON,
    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_dosha (user_id)
);
