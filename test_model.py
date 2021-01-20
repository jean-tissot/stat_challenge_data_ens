#créer autant de fonction qu'il y a de modèles différents à tester

def test_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)