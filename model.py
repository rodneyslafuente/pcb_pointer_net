from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder class for PCB Pointer Network
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initiate Encoder
        """

        super().__init__()

        # Input size is 2, given two coordinates per position
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)

    def forward(
        self, positions_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encoder forward pass

        :param Tensor positions: PCB component positions (batch_size, num_positions, 2)
        :return: LSTM return value (output, (hidden_states, cell_states))
        """

        return self.lstm(positions_batch)


class Attention(nn.Module):
    """
    Attention class for PCB Pointer Network
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initiate Attention
        """

        super().__init__()

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        Attention forward pass

        :param Tensor decoder_hidden: Decoder hidden state (batch_size, hidden_size)
        :param Tensor encoder_outputs: Encoder output (batch_size, components_size, hidden_size)
        :return: Tuple - (attention aware hidden state, attention)
        """

        D = self.W_1(decoder_hidden).unsqueeze(1)
        E = self.W_2(encoder_outputs)
        attention = torch.softmax(self.V(torch.tanh(D + E)), dim=1)

        attentioned_encoder_outputs = torch.mul(encoder_outputs, attention)
        atttention_aware_hidden_state = attentioned_encoder_outputs.sum(1)
        # (batch_size, components_size)
        attention = attention.squeeze(2)

        return atttention_aware_hidden_state, attention


class Decoder(nn.Module):
    """
    Decoder for PCB Pointer Network
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initiate Decoder
        """

        super().__init__()

        self.attention = Attention(hidden_size)

        # input_size is hidden_size + 2 given position coordinates
        self.lstm = nn.LSTM(hidden_size + 2, hidden_size, batch_first=True)

    def forward(
        self,
        position_batch: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Decoder forward pass

        :input Tensor position_batch: (batch_size, 1, 2)
        :input Tensor encoder_output: (batch_size, components_size, hidden_size)
        :input Tensor decoder_hidden: Tuple of LSTM (hidden state, cell state)
        :return: Tuple of LSTM hidden states, attention (batch_size, components_size)
        """

        # extract first LSTM's hidden state, not including cell state, to compute attention with
        hidden_state = decoder_hidden[0][0]

        attention_aware_hidden_state, attention = self.attention(
            hidden_state, encoder_output
        )
        attention_aware_hidden_state = attention_aware_hidden_state.unsqueeze(1)

        # (batch_size, 1, hidden_size + 2)
        lstm_input = torch.cat([attention_aware_hidden_state, position_batch], dim=2)

        # next_hidden is tuple - (hidden_state, cell_state)
        _, next_hidden = self.lstm(lstm_input, decoder_hidden)

        return next_hidden, attention


class PCBPointerNet(nn.Module):
    """
    PCB Pointer Network
    """

    def __init__(self, hidden_size: int) -> None:
        """
        Initiate Pointer Network
        """

        super().__init__()

        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, positions_batch: torch.Tensor) -> torch.Tensor:
        """
        Pointer Network forward pass

        :input Tensor positions: Batch of positions (batch_size, components_size, 2)
        :input Tensor orders: Bacth of orders (batch_size, components_size)
        :return: Pointers (batch_size, components_size, components_size)
        """

        components_size = positions_batch.shape[1]
        batch_size = positions_batch.shape[0]

        # encode positions
        encoder_output, hidden_states = self.encoder(positions_batch)

        # initialize decoder input
        decoder_input = torch.zeros((batch_size, 1, 2))

        pointers = torch.zeros((batch_size, 0, components_size))

        # decode encoder output
        for _ in range(components_size):
            hidden_states, attention = self.decoder(
                decoder_input, encoder_output, hidden_states
            )

            pointers = torch.cat([pointers, attention.unsqueeze(1)], dim=1)

            predictions = torch.argmax(attention, dim=1)

            # choose predictions from positions_batch as input for next decoder step
            for batch_num in range(batch_size):
                decoder_input[batch_num] = positions_batch[
                    batch_num, predictions[batch_num]
                ]

        return pointers
