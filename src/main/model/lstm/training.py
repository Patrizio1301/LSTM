

def training(model, X, y, epoche):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    model.fit(X, y, epochs=epoche, batch_size=32)
    return model

