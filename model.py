from transformers import BertModel
import torch.nn as nn
import torch
class Model_Text(nn.Module):
    def __init__(self, middle_layer_shape):
        super(Model_Text, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, middle_layer_shape)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, input, attention_mask):
        _, bert = self.bert(input, attention_mask = attention_mask)
        return self.dropout(self.relu(self.linear(bert)))

class LSTM_Model_With_Question(nn.Module):
    def __init__(self, num_classes, middle_layer_shape, hidden_size, device):
        super(LSTM_Model_With_Question, self).__init__()
        self.Forecast = Model_Text(middle_layer_shape)
        self.Question = Model_Text(middle_layer_shape)
        self.lstm = nn.LSTM((middle_layer_shape*2) + 1, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.dropout = nn.Dropout(0.5)
    def forward(self, input, attention, forecast_prediction, question_input, question_attention):
        maximum_length = input.size(1)
        cumulative_lstm_input = []
        question_information = self.Question(question_input[:, 0, :], question_attention[:, 0, :])
        for i in range(maximum_length):
            forecast_information = self.Forecast(input[:, i, :], attention[:, i, :])
            lstm_input = torch.cat((forecast_information, question_information, forecast_prediction[:, i].unsqueeze(1)), dim=-1)
            cumulative_lstm_input.append(lstm_input.tolist())
        cumulative_lstm_input = torch.tensor(cumulative_lstm_input, dtype=torch.float, device = self.device)
        output, (last_hidden, last_cell) = self.lstm(cumulative_lstm_input)
        final_output = self.dropout(self.out(last_hidden))
        return final_output

