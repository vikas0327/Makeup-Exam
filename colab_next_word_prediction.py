import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# 1. Dataset Creation
# A diverse, meaningful dataset of 250+ sentences
data = """Artificial intelligence is transforming the modern world.
Machine learning can improve various industries significantly.
Deep learning models are based on neural networks.
Natural language processing enables computers to understand text.
Computer vision allows machines to interpret visual information.
Data science is a multidisciplinary field of study.
Big data analytics helps businesses make informed decisions.
Cloud computing provides on-demand access to computing resources.
Internet of things connects physical devices to the internet.
Cybersecurity protects computer systems from unauthorized access.
Software engineering is the application of engineering principles.
Web development involves building interactive websites and applications.
Mobile app development creates software for smartphones and tablets.
Quantum computing has the potential to solve complex problems.
Blockchain technology provides a secure and decentralized ledger.
Virtual reality creates immersive digital environments for users.
Augmented reality overlays digital information onto the real world.
Robotics involves the design and construction of automated machines.
Automation is changing the landscape of the global workforce.
Technology continues to evolve at an unprecedented rapid pace.
The future of work will require advanced digital skills.
Education is being revolutionized by online learning platforms.
Healthcare is benefiting from personalized medicine and AI.
Renewable energy is essential for a sustainable future.
Electric vehicles are becoming more popular around the world.
Space exploration aims to discover new knowledge about the universe.
Climate change is a pressing issue that requires global action.
Environmental conservation helps protect our planet's natural resources.
Sustainable development meets the needs of the present generation.
Economic growth should be balanced with social responsibility.
Financial literacy is important for making sound money decisions.
Investment strategies can help grow your wealth over time.
Stock markets fluctuate based on various economic indicators.
Entrepreneurship drives innovation and creates new business opportunities.
Leadership involves inspiring and guiding others toward a goal.
Effective communication is key to building strong relationships.
Teamwork is essential for achieving success in any project.
Problem solving requires critical thinking and creativity.
Time management helps you prioritize tasks and increase productivity.
Stress management is crucial for maintaining mental well-being.
A healthy diet provides the necessary nutrients for your body.
Regular exercise improves cardiovascular health and overall fitness.
Adequate sleep is vital for physical and mental recovery.
Mental health awareness is growing in our society today.
Mindfulness practices can help reduce anxiety and improve focus.
Yoga and meditation promote relaxation and inner peace.
Art and music are universal forms of human expression.
Literature allows us to explore different perspectives and experiences.
History teaches us valuable lessons about the human past.
Philosophy encourages us to question our assumptions and beliefs.
Science relies on empirical evidence and the scientific method.
Mathematics is the universal language of patterns and logic.
Physics seeks to understand the fundamental laws of nature.
Chemistry studies the composition and properties of matter.
Biology is the scientific study of life and living organisms.
Genetics explores how traits are inherited from one generation.
Astronomy is the study of celestial objects and phenomena.
Psychology is the scientific study of the human mind.
Sociology examines human society and social behavior.
Anthropology is the study of human cultures and societies.
Geography studies the physical features of the earth and atmosphere.
Political science analyzes systems of government and political behavior.
Economics focuses on the production and consumption of goods.
Linguistics is the scientific study of language and its structure.
Law provides a framework for resolving disputes and maintaining order.
Justice is a fundamental principle of a fair and equitable society.
Human rights are inherent to all human beings everywhere.
Democracy is a system of government by the whole population.
Freedom of speech is a cornerstone of democratic societies.
Equality and diversity promote a more inclusive and tolerant world.
Cultural heritage connects us to our roots and shared history.
Travel allows us to experience different cultures and landscapes.
Food is a reflection of a region's history and geography.
Cooking is both an art and a science of preparation.
Sports and recreation bring people together and promote physical activity.
The Olympic Games celebrate athletic excellence and international cooperation.
Music has the power to evoke strong emotions and memories.
Movies and television are popular forms of visual entertainment.
Video games offer interactive and engaging storytelling experiences.
Photography captures moments in time and preserves visual memories.
Architecture shapes the physical environment in which we live.
Design combines aesthetics and functionality to solve specific problems.
Fashion is a dynamic and ever-changing form of self-expression.
Social media has transformed how we communicate and share information.
The internet is a vast repository of human knowledge.
Search engines help us find information quickly and easily.
Email remains a primary method of digital communication.
E-commerce has revolutionized the retail industry worldwide.
Online banking provides convenient access to financial services.
Digital marketing strategies help businesses reach their target audience.
Content creation is a growing industry in the digital age.
Artificial intelligence will continue to shape our future.
Machine learning algorithms are becoming more sophisticated and accurate.
Deep learning networks can process vast amounts of complex data.
Natural language models can generate human-like text responses.
Computer vision systems can recognize faces and objects accurately.
Data scientists are in high demand across many industries.
Big data requires advanced storage and processing capabilities.
Cloud platforms offer scalable and flexible infrastructure solutions.
Internet of things devices generate massive streams of data.
Cyber threats are becoming more frequent and sophisticated.
Software developers write the code that powers our digital world.
Web designers focus on creating intuitive user interfaces.
Mobile applications have changed how we access information.
Quantum computers will solve problems that are currently intractable.
Blockchain can be used for more than just cryptocurrencies.
Virtual reality has applications in training and simulation.
Augmented reality can enhance our shopping and learning experiences.
Robots are increasingly used in manufacturing and healthcare.
Automation will displace some jobs but create new ones.
Technological progress presents both opportunities and significant challenges.
The transition to renewable energy is accelerating globally.
Electric cars produce zero tailpipe emissions during operation.
Solar and wind power are key components of clean energy.
Battery technology is improving energy storage capacity.
Space agencies are planning missions to Mars and beyond.
Commercial spaceflight is becoming a reality for private citizens.
Satellites provide essential navigation and communication services.
Climate models help us predict future environmental changes.
Conservation efforts aim to protect endangered species and habitats.
Sustainable agriculture practices help preserve soil and water quality.
The circular economy aims to eliminate waste and pollution.
Global trade connects economies and cultures around the world.
Supply chains are complex networks that deliver goods globally.
Inflation and interest rates affect consumer spending and investment.
Central banks play a crucial role in managing the economy.
Venture capital funds innovative startups and new technologies.
Startups disrupt traditional industries with new ideas and products.
Corporate social responsibility is increasingly important to consumers.
Ethics in AI is a growing area of concern and research.
Data privacy and security are fundamental user rights.
Algorithmic bias can lead to unfair or discriminatory outcomes.
Transparency and accountability are essential for trustworthy AI.
Lifelong learning is necessary to adapt to a changing world.
Critical thinking skills help us evaluate information objectively.
Emotional intelligence is important for effective leadership and teamwork.
Empathy allows us to understand and share the feelings of others.
Resilience helps us bounce back from adversity and challenges.
Adaptability is a key trait for success in the modern workplace.
Creativity is the ability to generate novel and useful ideas.
Innovation drives progress and improves our quality of life.
The pursuit of knowledge is a lifelong journey of discovery.
Curiosity is the engine of intellectual growth and learning.
Passion and purpose give meaning to our personal and professional lives.
Goal setting helps us focus our efforts and track progress.
Habits play a significant role in shaping our daily routines.
Motivation can be intrinsic or extrinsic in nature.
Discipline is required to achieve long-term goals and success.
Self-awareness is the foundation of personal growth and development.
Feedback is essential for continuous improvement and learning.
Mentorship can provide valuable guidance and support.
Networking helps build professional relationships and opportunities.
Collaboration often leads to better outcomes than working alone.
Diversity of thought leads to more innovative solutions.
Inclusion ensures that everyone has a voice and an opportunity.
A positive attitude can improve your overall well-being.
Gratitude helps us appreciate the good things in our lives.
Optimism is associated with better physical and mental health.
Laughter is a powerful antidote to stress and pain.
Play is important for creativity and cognitive development.
Nature has a calming and restorative effect on the human mind.
Spending time outdoors improves physical and mental health.
Walking is a simple and effective form of exercise.
Reading expands your vocabulary and broadens your perspective.
Writing is a powerful tool for organizing your thoughts.
Journaling can help you process your emotions and experiences.
Public speaking is a valuable skill in many professions.
Active listening improves communication and builds trust.
Negotiation is a process of reaching a mutually beneficial agreement.
Conflict resolution requires patience and understanding.
Decision making involves weighing the pros and cons of different options.
Risk management is essential for minimizing potential losses.
Strategic planning helps organizations achieve their long-term objectives.
Project management ensures that tasks are completed on time.
Quality assurance ensures that products meet specified standards.
Customer service is crucial for building brand loyalty.
User experience design focuses on the needs of the end user.
Agile methodologies promote iterative and collaborative software development.
Open source software encourages community contributions and sharing.
The developer community is vibrant and constantly evolving.
Continuous integration and deployment streamline the software release process.
Version control systems help track changes to source code.
Debugging is the process of finding and fixing software errors.
Testing ensures that software functions as expected.
Documentation is important for understanding how a system works.
Refactoring improves the structure and readability of existing code.
Code reviews help catch errors and improve code quality.
Best practices provide guidelines for effective and efficient development.
Design patterns offer reusable solutions to common programming problems.
Algorithms are step-by-step instructions for solving a problem.
Data structures organize and store data efficiently.
Databases store and retrieve large amounts of structured information.
SQL is a standard language for managing relational databases.
NoSQL databases offer flexible schemas for unstructured data.
APIs allow different software systems to communicate with each other.
Microservices architecture breaks down applications into smaller independent components.
Serverless computing allows developers to focus on code rather than infrastructure.
Containerization packages applications with their dependencies.
Docker and Kubernetes are popular tools for container management.
The command line is a powerful interface for interacting with computers.
Linux is a widely used open-source operating system.
Networking protocols define how data is transmitted over the internet.
TCP/IP is the foundational protocol suite of the internet.
HTTP is the protocol used for transmitting web pages.
SSL and TLS protocols provide secure communication over a network.
Encryption transforms data into an unreadable format for security.
Authentication verifies the identity of a user or system.
Authorization determines what actions a user is permitted to perform.
Two-factor authentication adds an extra layer of security.
Malware is malicious software designed to harm or exploit systems.
Phishing is a common social engineering attack to steal credentials.
Firewalls monitor and control incoming and outgoing network traffic.
Antivirus software helps detect and remove malicious programs.
Regular backups are essential for recovering from data loss.
System administration involves managing and maintaining computer systems.
DevOps practices integrate software development and IT operations.
Site reliability engineering focuses on ensuring system availability and performance.
Monitoring and logging are crucial for diagnosing system issues.
Scalability is the ability of a system to handle increased load.
Performance optimization improves the speed and efficiency of a system.
Artificial intelligence models require extensive training data.
The quality of data directly impacts the performance of machine learning models.
Data cleaning and preprocessing are essential steps in the data pipeline.
Feature engineering involves selecting and transforming relevant variables.
Hyperparameter tuning optimizes the configuration of a machine learning model.
Cross-validation helps evaluate the generalization performance of a model.
Overfitting occurs when a model learns the training data too well.
Underfitting occurs when a model fails to capture the underlying patterns.
Regularization techniques help prevent overfitting in machine learning models.
Ensemble methods combine multiple models to improve predictive accuracy.
Transfer learning leverages knowledge from pre-trained models.
Reinforcement learning agents learn by interacting with their environment.
Generative adversarial networks can create realistic synthetic data.
Recurrent neural networks are well-suited for sequential data processing.
Convolutional neural networks are highly effective for image recognition tasks.
Transformer architecture has revolutionized natural language processing.
Attention mechanisms allow models to focus on relevant parts of the input.
Pre-trained language models have achieved state-of-the-art results.
Fine-tuning adapts a pre-trained model to a specific task.
Ethical considerations must be integrated into the development of AI systems.
Explainable AI aims to make machine learning models more transparent.
The democratization of AI tools is empowering more people to build solutions.
The intersection of AI and healthcare holds immense promise.
AI can accelerate drug discovery and development processes.
Personalized learning powered by AI can improve educational outcomes.
Smart homes use IoT devices to enhance comfort and energy efficiency.
Autonomous drones have applications in delivery and surveillance.
Precision agriculture uses technology to optimize crop yields.
The gig economy is transforming the traditional employment model.
Remote work has become more prevalent and acceptable.
Digital nomads leverage technology to work from anywhere in the world.
Coworking spaces provide flexible office environments for freelancers and remote workers.
The balance between work and life is essential for long-term happiness.
Prioritizing mental health is crucial in a fast-paced world.
Self-care is not selfish but necessary for well-being.
Building a supportive community is important for personal and professional growth.
Continuous adaptation is the key to thriving in the twenty-first century.
The future belongs to those who are willing to learn and innovate."""

def predict_next_word(model, tokenizer, max_sequence_len, text, top_k=3):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return []
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    
    top_words = []
    for index in top_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_words.append((word, predicted_probs[index]))
                break
    return top_words

def auto_complete_sentence(model, tokenizer, max_sequence_len, text, num_words=5):
    current_text = text
    for _ in range(num_words):
        top_words = predict_next_word(model, tokenizer, max_sequence_len, current_text, top_k=1)
        if top_words:
            next_word = top_words[0][0]
            current_text += " " + next_word
        else:
            break
    return current_text

def main():
    print("Starting Next Word Prediction Training...\n")
    
    # 2. NLP Preprocessing
    corpus = data.lower().split("\n")
    corpus = [line.strip() for line in corpus if line.strip() != ""]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    print(f"Total vocabulary size: {total_words}")

    # Sequence generation
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Padding sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    print(f"\nSample sequences (first 5):\n{input_sequences[:5]}")
    print(f"\nMaximum sequence length: {max_sequence_len}")

    # Create Predictors and Label
    X, labels = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    # 3. LSTM Model Development
    print("\nBuilding LSTM model...")
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        LSTM(150),
        Dense(total_words, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    print("\nTraining the model for 100 epochs...")
    history = model.fit(X, y, epochs=100, verbose=1)

    # 4. Performance Visualization
    print("\nGenerating performance visualizations...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

    # 5. Interactive Prediction Interface
    print("\n" + "="*50)
    print("Model Training Complete! You can now test predictions.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter a sentence fragment (or type 'exit' to quit): ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
            
        print("\nPredicting...")
        # Get top 3 words
        top_predictions = predict_next_word(model, tokenizer, max_sequence_len, user_input, top_k=3)
        
        if top_predictions:
            print(f"Top 3 Next Word Predictions:")
            for i, (word, prob) in enumerate(top_predictions, 1):
                print(f"{i}. {word} (Confidence: {prob:.4f})")
            
            print(f"\nPrediction (Highest Confidence): '{top_predictions[0][0]}'")
            
            # Auto-completion
            completed = auto_complete_sentence(model, tokenizer, max_sequence_len, user_input, num_words=3)
            print(f"Auto-completion suggestion: {completed}")
        else:
            print("Could not generate a prediction. Please try a different phrase.")

if __name__ == "__main__":
    main()
